/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/npu_torch/dcp_attention_utils.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace xllm::layer::test {
namespace {

std::pair<torch::Tensor, torch::Tensor> compute_attention_partial(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.sizes(), key.sizes());
  CHECK_EQ(query.size(1), key.size(1));
  CHECK_EQ(query.size(2), key.size(2));

  const double scale = 1.0 / std::sqrt(static_cast<double>(query.size(2)));
  const torch::Tensor scores =
      torch::einsum("qhd,khd->qhk", {query, key}) * scale;
  const torch::Tensor partial_lse = torch::logsumexp(scores, -1, true);
  const torch::Tensor partial_out =
      torch::einsum("qhk,khd->qhd", {torch::softmax(scores, -1), value});
  return {partial_out, partial_lse};
}

TEST(DcpAttentionUtilsTest, ShardedKvMergeMatchesFullAttentionInFloat64) {
  torch::manual_seed(7);
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
  const torch::Tensor query = torch::randn({5, 3, 8}, options);
  const torch::Tensor key = torch::randn({12, 3, 8}, options);
  const torch::Tensor value = torch::randn({12, 3, 8}, options);
  const auto [reference_out, reference_lse] =
      compute_attention_partial(query, key, value);
  (void)reference_lse;

  const std::vector<torch::Tensor> key_shards = key.chunk(3, 0);
  const std::vector<torch::Tensor> value_shards = value.chunk(3, 0);
  ASSERT_EQ(key_shards.size(), value_shards.size());
  std::vector<torch::Tensor> partial_outputs;
  std::vector<torch::Tensor> partial_lses;
  partial_outputs.reserve(key_shards.size());
  partial_lses.reserve(key_shards.size());
  for (int64_t shard_index = 0;
       shard_index < static_cast<int64_t>(key_shards.size());
       ++shard_index) {
    const auto [partial_out, partial_lse] = compute_attention_partial(
        query, key_shards[shard_index], value_shards[shard_index]);
    partial_outputs.emplace_back(partial_out);
    partial_lses.emplace_back(partial_lse);
  }

  const torch::Tensor merged_out = detail::merge_dcp_partials(
      torch::stack(partial_outputs, 0), torch::stack(partial_lses, 0));
  EXPECT_TRUE(torch::allclose(
      merged_out, reference_out, /*rtol=*/1e-10, /*atol=*/1e-10));
}

TEST(DcpAttentionUtilsTest, InvalidLseShardIsIgnored) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
  const torch::Tensor valid_out =
      torch::tensor({3.0, 7.0}, options).view({1, 2, 1});
  const torch::Tensor invalid_out =
      torch::full_like(valid_out, std::numeric_limits<double>::quiet_NaN());
  const torch::Tensor valid_lse = torch::zeros({1, 2, 1}, options);
  const torch::Tensor invalid_lse =
      torch::full_like(valid_lse, -std::numeric_limits<double>::infinity());

  const torch::Tensor merged_out =
      detail::merge_dcp_partials(torch::stack({invalid_out, valid_out}, 0),
                                 torch::stack({invalid_lse, valid_lse}, 0));
  EXPECT_TRUE(torch::equal(merged_out, valid_out));
}

TEST(DcpAttentionUtilsTest, AllInvalidLseShardsProduceZero) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
  const torch::Tensor partial_out = torch::full(
      {4, 2, 3, 5}, std::numeric_limits<double>::quiet_NaN(), options);
  const torch::Tensor partial_lse = torch::full(
      {4, 2, 3, 1}, -std::numeric_limits<double>::infinity(), options);

  const torch::Tensor merged_out =
      detail::merge_dcp_partials(partial_out, partial_lse);
  EXPECT_TRUE(torch::equal(merged_out, torch::zeros_like(merged_out)));
}

TEST(DcpAttentionUtilsTest, DistributesPartialTailAcrossFourRanks) {
  std::vector<int64_t> local_kv_seq_lens;
  local_kv_seq_lens.reserve(4);
  for (int32_t dcp_rank = 0; dcp_rank < 4; ++dcp_rank) {
    const std::vector<int64_t> rank_local_kv_seq_lens =
        detail::compute_dcp_local_kv_seq_lens(
            /*global_kv_seq_lens=*/{257},
            /*dcp_size=*/4,
            dcp_rank,
            /*block_size=*/128);
    ASSERT_EQ(rank_local_kv_seq_lens.size(), 1);
    local_kv_seq_lens.emplace_back(rank_local_kv_seq_lens.front());
  }

  EXPECT_EQ(local_kv_seq_lens, (std::vector<int64_t>{128, 128, 1, 0}));
}

}  // namespace
}  // namespace xllm::layer::test

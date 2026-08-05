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

#include <algorithm>
#include <tuple>

namespace xllm::layer::detail {

std::vector<int64_t> compute_dcp_local_kv_seq_lens(
    const std::vector<int64_t>& global_kv_seq_lens,
    int32_t dcp_size,
    int32_t dcp_rank,
    int64_t block_size) {
  CHECK_GT(dcp_size, 1);
  CHECK_GE(dcp_rank, 0);
  CHECK_LT(dcp_rank, dcp_size);
  CHECK_GT(block_size, 0);

  std::vector<int64_t> local_kv_seq_lens;
  local_kv_seq_lens.reserve(global_kv_seq_lens.size());
  for (const int64_t global_kv_seq_len : global_kv_seq_lens) {
    CHECK_GE(global_kv_seq_len, 0);
    const int64_t base = global_kv_seq_len / block_size / dcp_size * block_size;
    const int64_t remainder = global_kv_seq_len - base * dcp_size;
    const int64_t rank_offset = static_cast<int64_t>(dcp_rank) * block_size;
    const int64_t local_remainder =
        std::clamp(remainder - rank_offset, int64_t{0}, block_size);
    local_kv_seq_lens.emplace_back(base + local_remainder);
  }
  return local_kv_seq_lens;
}

torch::Tensor merge_dcp_partials(const torch::Tensor& all_partial_out,
                                 const torch::Tensor& all_partial_lse) {
  CHECK(all_partial_out.scalar_type() == torch::kFloat32 ||
        all_partial_out.scalar_type() == torch::kFloat64);
  CHECK_EQ(all_partial_lse.scalar_type(), all_partial_out.scalar_type());
  CHECK_EQ(all_partial_out.dim(), 4);
  CHECK_EQ(all_partial_lse.dim(), 4);
  CHECK_EQ(all_partial_out.size(0), all_partial_lse.size(0));
  CHECK_EQ(all_partial_out.size(1), all_partial_lse.size(1));
  CHECK_EQ(all_partial_out.size(2), all_partial_lse.size(2));
  CHECK_EQ(all_partial_lse.size(3), 1);

  const torch::Tensor finite_lse = torch::isfinite(all_partial_lse);
  const torch::Tensor max_lse = std::get<0>(all_partial_lse.max(0));
  const torch::Tensor max_lse_is_finite = torch::isfinite(max_lse);
  const torch::Tensor safe_max_lse =
      torch::where(max_lse_is_finite, max_lse, torch::zeros_like(max_lse));
  const torch::Tensor weights =
      torch::where(finite_lse,
                   torch::exp(all_partial_lse - safe_max_lse),
                   torch::zeros_like(all_partial_lse));
  const torch::Tensor safe_partial_out =
      torch::where(finite_lse.expand_as(all_partial_out),
                   all_partial_out,
                   torch::zeros_like(all_partial_out));
  const torch::Tensor denominator = weights.sum(0);
  const torch::Tensor safe_denominator = torch::where(
      denominator.gt(0), denominator, torch::ones_like(denominator));
  const torch::Tensor merged_out =
      (weights * safe_partial_out).sum(0) / safe_denominator;
  return torch::where(max_lse_is_finite.expand_as(merged_out),
                      merged_out,
                      torch::zeros_like(merged_out));
}

}  // namespace xllm::layer::detail

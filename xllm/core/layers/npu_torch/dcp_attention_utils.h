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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm::layer::detail {

std::vector<int64_t> compute_dcp_local_kv_seq_lens(
    const std::vector<int64_t>& global_kv_seq_lens,
    int32_t dcp_size,
    int32_t dcp_rank,
    int64_t block_size);

torch::Tensor merge_dcp_partials(const torch::Tensor& all_partial_out,
                                 const torch::Tensor& all_partial_lse);

}  // namespace xllm::layer::detail

#pragma once
#include <torch/torch.h>

torch::Tensor create_2d_tensor(unsigned int row, unsigned int column);
torch::Tensor mat_mul(torch::Tensor tensor1, torch::Tensor tensor2);
torch::Tensor ele_mul(torch::Tensor tensor1, torch::Tensor tensor2);
torch::Tensor reshape_tensor(torch::Tensor tensor, const std::array<int64_t,2>& new_shape);
torch::Tensor transpose_tensor(torch::Tensor tensor);

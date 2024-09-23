#include "../include/basic_tensor.h"
#include <ATen/core/TensorBody.h>
#include <utility>

torch::Tensor create_2d_tensor(unsigned int row, unsigned int column) {
    // Create a random tensor with the given dimensions (row x column)
    torch::Tensor tensor = torch::rand({static_cast<long>(row), static_cast<long>(column)});
    return tensor;
}

torch::Tensor mat_mul(torch::Tensor tensor1, torch::Tensor tensor2) {
    // Multiply two tensors
    torch::Tensor result = torch::matmul(tensor1, tensor2);
    return result;
}

torch::Tensor ele_mul(torch::Tensor tensor1, torch::Tensor tensor2) {
    // Element-wise multiplication of two tensors
    torch::Tensor result = tensor1 * tensor2;
    return result;
}

torch::Tensor reshape_tensor(torch::Tensor tensor, const std::array<int64_t, 2>& new_shape) {
    // Reshape the tensor to the new shape
    torch::Tensor result = tensor.reshape(new_shape);
    return result;
}

torch::Tensor transpose_tensor(torch::Tensor tensor) {
    // Transpose the tensor
    torch::Tensor result = tensor.t();
    return result;
}
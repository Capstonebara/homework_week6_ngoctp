#include <iostream>
#include <torch/torch.h>

#include "neural_network.h"
#include "basic_tensor.h"

int main() {
    int option;

    std::cout << "Choose an option: " << std::endl;
    std::cout << "1. Create a random 2d tensor" << std::endl;
    std::cout << "2. Multiply two tensors" << std::endl;
    std::cout << "3. Element-wise multiplication of two tensors" << std::endl;
    std::cout << "4. Reshape a tensor" << std::endl;
    std::cout << "5. Transpose a tensor" << std::endl;
    std::cout << "6. Create a feedforward neural network model" << std::endl;

    std::cin >> option;

    switch (option) {
    case 1: {
        // create a tensor
        torch::Tensor tensor = create_2d_tensor(6 ,6).cuda();
        std::cout << "Random 2d tensor" << std::endl;
        std::cout << tensor << std::endl;
        break;
    }

    case 2: {
        // matmul tensor
        torch::Tensor tensor1 = torch::tensor({{1,2}, {3,4}, {5,6}});
        torch::Tensor tensor2 = torch::tensor({{23,6, 2003}, {21, 4, 2003}});
    torch::Tensor matmul_tensor = mat_mul(tensor1, tensor2).cuda();
        std::cout << "tensor1:" << std::endl;
        std::cout << tensor1 << std::endl;
        std::cout << "tensor2:" << std::endl;
        std::cout << tensor2 << std::endl;
        std::cout << "Matmul tensor" << std::endl;
        std::cout << matmul_tensor << std::endl;
        break;
    }

    case 3: {
        // element-wise multiplication tensor
        torch::Tensor tensorA = torch::tensor({{1,2}, {3,4}, {5,6}});
        torch::Tensor tensorB = torch::tensor({{23,6}, {21, 4}, {2003, 2003}});
        torch::Tensor ele_mul_tensor = ele_mul(tensorA, tensorB).cuda();
        std::cout << "tensorA:" << std::endl;
        std::cout << tensorA << std::endl;
        std::cout << "tensorB:" << std::endl;
        std::cout << tensorB << std::endl;
        std::cout << "Element-wise multiplication tensor" << std::endl;
        std::cout << ele_mul_tensor << std::endl;
        break;
    }
    case 4: {
        // reshape tensor
        torch::Tensor org_tensor = torch::tensor({{1}, {2}, {3}, {4}, {5}, {6}});
        torch::Tensor reshaped_tensor = reshape_tensor(org_tensor, {2, 3}).cuda();
        std::cout << "Original tensor:" << std::endl;
        std::cout << org_tensor << std::endl;
        std::cout << "Reshape tensor" << std::endl;
        std::cout << reshaped_tensor << std::endl;
        break;
    }   

    case 5: {
        // transpose tensor
        torch::Tensor org_tensor2 = torch::tensor({{1,2}, {3,4}, {5,6}});
        torch::Tensor transposed_tensor = transpose_tensor(org_tensor2).cuda();
        std::cout << "Original tensor:" << std::endl;
        std::cout << org_tensor2 << std::endl;
        std::cout << "Transpose tensor" << std::endl;
        std::cout << transposed_tensor << std::endl;
        break;
    }

    case 6: {
        // Create a feedforward neural network model
        float target_weight = 6.0;
        float target_bias = 4.0;
        float weight, bias;
        auto [learned_weight, learned_bias] = Execute(target_weight, target_bias);
        std::cout << "Target weight: " << target_weight << std::endl;
        std::cout << "Target bias: " << target_bias << std::endl;
        std::cout << "Learned weight: " << learned_weight << std::endl;
        std::cout << "Learned bias: " << learned_bias << std::endl;
        break;
    }

    default:
		std::cout << "Not support" << std::endl;
		break;
    }

    return 0;
}


#ifndef FEEDFORWARD_MODEL_H
#define FEEDFORWARD_MODEL_H

#include <torch/torch.h>


// Define the FeedforwardNeuralNetModel structure
struct FeedforwardNeuralNetModel : torch::nn::Module {
    torch::nn::Linear fc{nullptr};
    // torch::nn::ReLU relu;

    FeedforwardNeuralNetModel(int input_dim, int output_dim) {
        fc = register_module("fc1", torch::nn::Linear(input_dim, output_dim));
        // fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = fc->forward(x);
        // x = relu->forward(x);
        // x = fc2->forward(x);
        return x;
    }
};

// Function declaration for Execute()
std::pair<torch::Tensor, torch::Tensor> Execute(float target_weight, float target_bias);

#endif // FEEDFORWARD_MODEL_H


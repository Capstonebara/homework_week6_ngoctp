#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "../include/neural_network.h"

std::pair<torch::Tensor, torch::Tensor> Execute(float target_weight, float target_bias) {
    torch::Device device = torch::Device(torch::kCUDA);
    int input_size = 1;
    // int hidden_size = 5;
    int output_size = 1;
    int epochs = 1000;
    double learning_rate = 0.1;

    // Initialize the model, loss function, and optimizer
    auto model = std::make_shared<FeedforwardNeuralNetModel>(input_size, output_size);
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    torch::nn::MSELoss criterion;

    // Create data
    double start = 0.0;
    double end = 1.0;
    double step = 0.02;
    auto X = torch::arange(start, end, step).unsqueeze(1).to(device);
    auto y = (target_weight * X + target_bias).to(device);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        auto outputs = model->forward(X);
        auto loss = criterion(outputs, y);

        // Backward pass and optimization
        optimizer.zero_grad(); 
        loss.backward();
        optimizer.step();

        // Print the loss every 10 epochs
        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "], Loss: " << loss.item<double>() << std::endl;
            // auto predicted = model->forward(X);
            // std::cout << "Predicted outputs:\n" << predicted << std::endl;
        }
    }

    // Get the weight and bias of the output layer (fc2)
    torch::Tensor learned_weight = model->fc->weight.data();
    torch::Tensor learned_bias = model->fc->bias.data();

    return std::make_pair(learned_weight, learned_bias);
}
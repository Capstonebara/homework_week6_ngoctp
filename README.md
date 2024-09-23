# 📚 Homework Week 6 - NgocTP

## 📝 Overview
This repository contains the code and resources for Homework Week 6 of the Capstonebara project. The project is written in C++, uses CMake for building, and Makefile for additional configurations. **This project is configured to use CUDA for GPU acceleration, rather than running on the CPU.**

## 🏗️ Structure
- **🛠️ build/**: Contains build-related files.
- **📂 include/**: Header files.
- **💻 src/**: Source files.
- **📄 CMakeLists.txt**: CMake configuration file.
- **📝 README.md**: This file.

## 🚀 Getting Started
### ✅ Prerequisites
- C++ compiler
- CMake
- Make
- **CUDA toolkit and compatible GPU**

### 🏗️ Building the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/Capstonebara/homework_week6_ngoctp.git
   ```
2. Navigate to the project directory:
   ```sh
   cd homework_week6_ngoctp
   ```
3. Create a build directory and navigate into it:
   ```sh
   mkdir build && cd build
   ```
4. Make sure to configure the path of the **CUDA compiler**, **NVTOOLSEXT library** and **libtorch library**.
![Alt text](image.png)

5. Run CMake to configure the project with CUDA:
   ```sh
   cmake ..
   ```
6. Build the project using Make:
   ```sh
   make
   ```

Certainly! Here’s the combined "Usage" section for your README, including both the tensor utility functions and the feedforward neural network model with the `Execute` function:

---

## ⚙️ Usage

This project provides several utility functions for tensor operations and a simple feedforward neural network model for regression tasks using PyTorch. Below are the available functions and their descriptions.

![Alt text](image-1.png)

### Tensor Utility Functions

1. **`torch::Tensor create_2d_tensor(unsigned int row, unsigned int column)`**
   - **Description**: Creates a 2-dimensional tensor filled with random values.
   - **Parameters**:
     - `row`: The number of rows in the tensor.
     - `column`: The number of columns in the tensor.
   - **Returns**: A tensor of shape `(row, column)` filled with random values.

   **Example**:
   ```cpp
   torch::Tensor my_tensor = create_2d_tensor(3, 4); // Creates a 3x4 tensor
   ```

2. **`torch::Tensor mat_mul(torch::Tensor tensor1, torch::Tensor tensor2)`**
   - **Description**: Performs matrix multiplication between two tensors.
   - **Parameters**:
     - `tensor1`: The first tensor (should be 2D).
     - `tensor2`: The second tensor (should be 2D, with compatible dimensions for multiplication).
   - **Returns**: A tensor that is the result of the matrix multiplication.

   **Example**:
   ```cpp
   torch::Tensor result = mat_mul(tensor1, tensor2); // Multiplies tensor1 and tensor2
   ```

3. **`torch::Tensor ele_mul(torch::Tensor tensor1, torch::Tensor tensor2)`**
   - **Description**: Performs element-wise multiplication of two tensors.
   - **Parameters**:
     - `tensor1`: The first tensor.
     - `tensor2`: The second tensor (must have the same shape as tensor1).
   - **Returns**: A tensor that contains the element-wise product of `tensor1` and `tensor2`.

   **Example**:
   ```cpp
   torch::Tensor result = ele_mul(tensor1, tensor2); // Performs element-wise multiplication
   ```

4. **`torch::Tensor reshape_tensor(torch::Tensor tensor, const std::array<int64_t, 2>& new_shape)`**
   - **Description**: Reshapes a tensor to a specified new shape.
   - **Parameters**:
     - `tensor`: The tensor to be reshaped.
     - `new_shape`: An array specifying the new shape of the tensor (must have the same number of elements as the original tensor).
   - **Returns**: A reshaped tensor with the dimensions defined by `new_shape`.

   **Example**:
   ```cpp
   std::array<int64_t, 2> new_shape = {2, 6}; // New shape
   torch::Tensor reshaped_tensor = reshape_tensor(original_tensor, new_shape); // Reshapes to 2x6
   ```

5. **`torch::Tensor transpose_tensor(torch::Tensor tensor)`**
   - **Description**: Transposes the given tensor, swapping its dimensions.
   - **Parameters**:
     - `tensor`: The tensor to be transposed (should be 2D).
   - **Returns**: A tensor that is the transposed version of the input tensor.

   **Example**:
   ```cpp
   torch::Tensor transposed_tensor = transpose_tensor(tensor); // Transposes the tensor
   ```

### Feedforward Neural Network Model

1. **`struct FeedforwardNeuralNetModel`**
   - **Description**: Defines a feedforward neural network model with one linear layer.
   - **Members**:
     - `torch::nn::Linear fc`: A fully connected (linear) layer that takes an input dimension and outputs a specified dimension.

   - **Constructor**:
     - `FeedforwardNeuralNetModel(int input_dim, int output_dim)`: Initializes the model by creating a linear layer with the given input and output dimensions.

   - **Method**:
     - `torch::Tensor forward(torch::Tensor x)`: Performs the forward pass through the network and returns the output.

   **Example**:
   ```cpp
   FeedforwardNeuralNetModel model(input_size, output_size);
   torch::Tensor output = model.forward(input_tensor); // Forward pass
   ```

2. **`std::pair<torch::Tensor, torch::Tensor> Execute(float target_weight, float target_bias)`**
   - **Description**: Trains the feedforward neural network model on a linear regression problem defined by the target weight and bias.
   - **Parameters**:
     - `target_weight`: The weight of the target linear function.
     - `target_bias`: The bias of the target linear function.
   - **Returns**: A pair of tensors containing the learned weight and bias of the output layer after training.

   **Training Process**:
   - Initializes the feedforward neural network model with input and output dimensions set to 1.
   - Creates a dataset by generating input data (`X`) in the range [0, 1) and calculating the corresponding output data (`y`) using the equation \(y = \text{target\_weight} \times X + \text{target\_bias}\).
   - Sets up the Mean Squared Error (MSE) loss function and the Stochastic Gradient Descent (SGD) optimizer.
   - Trains the model over a specified number of epochs, performing forward passes, loss calculations, and backpropagation.
   - Outputs the training loss every 100 epochs.
   - Returns the learned weight and bias of the model.

   **Example**:
   ```cpp
   float target_weight = 2.0f; // Example weight
   float target_bias = 0.5f;   // Example bias
   auto [learned_weight, learned_bias] = Execute(target_weight, target_bias);
   std::cout << "Learned weight: " << learned_weight << ", Learned bias: " << learned_bias << std::endl;
   ```


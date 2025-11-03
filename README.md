# A Concurrent Neural Network in Go

A from-scratch, fully concurrent neural network simulator in Go — each neuron runs two goroutines (one for feedforward, one for backprop) and communicates via channels, mimicking asynchronous biological signalling.

This project is primarily an educational experiment to explore how Go's concurrency primitives can be used to model the parallel nature of a neural network's architecture.

## Features

* **Concurrent Architecture:** Each neuron operates as an independent Goroutine, communicating with other neurons (and layers) using Go channels for inputs, outputs, and error signals.
* **Backpropagation:** Implements the standard backpropagation algorithm for training, fully utilizing the channel-based architecture.
* **Weight Persistence:** Includes functionality to save and load trained weights using a JSON file (`weights.json`).
* **Data Handling:** Provides a utility function (`LoadCSV`) for loading training data from a CSV, including automatic handling and **one-hot encoding** of categorical features.
* **Loss Functions:** Supports **Mean Squared Error (MSE)** for regression tasks (default) and **Cross-Entropy Loss** for classification (when `useCrossEntropy = true`).

## Getting Started

### Prerequisites

* Go (version 1.18 or later recommended)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/vishalvp-109989/go-concurrent-nn.git
    cd go-concurrent-nn
    go mod tidy
    ```

2.  Acquire the sample data (e.g., a CSV named `data.csv`).

3.  Run the training process:
    ```bash
    go run .
    ```

## Configuration

Key parameters can be adjusted directly in `main.go`:

| Constant | Description | Default Value |
| :--- | :--- | :--- |
| `learning_rate` | Controls the step size for weight updates. | `0.0001` |
| `useCrossEntropy` | Toggles between **MSE (false)** and **Cross-Entropy (true)** loss. | `false` |
| `K_CLASSES` | Number of classes for classification (only used with Cross-Entropy). | `1` |
| `EPOCH` | Number of full passes over the training dataset. | `100` |
| `BATCH` | Channel buffer size, acts as a micro-batching mechanism. | `1` |

The network structure is defined in `main.go`:

```go
network := NewNetwork(
    Dense(16, inputDim), // Input dimension setup
    Dense(8),    // First hidden layer with 8 neurons
    Dense(4),    // Second hidden layer with 4 neurons
    Dense(1),    // Output layer with 1 neuron (for regression)
)

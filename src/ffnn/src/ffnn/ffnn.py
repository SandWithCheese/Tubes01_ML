from tqdm.auto import tqdm
from ffnn.layer import Layer
from ffnn.types import ActivationFunction, LossFunction, WeightsSetup, WeightInitializer
from ffnn.activation import Activation
from ffnn.loss import Loss
from random import randint
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

class FFNN:
    def __init__(
        self,
        layer_sizes: list[int],
        activation_functions: list[ActivationFunction] = None,
        loss_function: LossFunction = LossFunction.MEAN_SQUARED_ERROR,
        weights_setup: list[WeightsSetup] = None,
        learning_rate: float = 0.0001,
        batch_size: int = 256,
        epochs: int = 100,
        verbose: bool = False,
        random_state: int = None,
    ):
        if activation_functions is None:
            activation_functions = [ActivationFunction.RELU] * (
                len(layer_sizes) - 2
            ) + [ActivationFunction.SOFTMAX]

        if weights_setup is None:
            weights_setup = [WeightsSetup(WeightInitializer.XAVIER)] * (
                len(layer_sizes) - 1
            )

        if random_state is None:
            random_state = randint(0, 1000)

        assert (
            len(layer_sizes) == len(activation_functions) + 1
        ), "Number of activation functions should be one less than number of layers"

        assert (
            len(layer_sizes) == len(weights_setup) + 1
        ), "Number of weights setup should be one less than number of layers"

        assert all(
            activation_function
            in [
                ActivationFunction.LINEAR,
                ActivationFunction.RELU,
                ActivationFunction.SIGMOID,
                ActivationFunction.TANH,
                ActivationFunction.SOFTMAX,
            ]
            for activation_function in activation_functions
        ), "Invalid activation function"

        for i in range(len(weights_setup)):
            weights_setup[i].seed = random_state

        self.layers: list[Layer] = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(
                Layer(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                    activation_functions[i - 1],
                    weights_setup[i - 1],
                )
            )

        self.loss_function = Loss.get_loss_from_type(loss_function)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.train_losses = []
        self.val_losses = []
        self.weights_grad = []
        self.biases_grad = []

    def show_graph(self):
        '''
        Show the graph of the neural network including bias neurons
        '''
        # Initialize variables
        positions = []
        edges = []
        edge_labels = []
        labels = []
        layer_x_offset = 200
        neuron_y_offset = 100
        
        # Create positions for all neurons
        current_neuron_idx = 0
        
        # Input layer
        input_size = self.layers[0].input_size
        for j in range(input_size):
            x = 0
            y = (j - input_size / 2) * neuron_y_offset
            positions.append((x, y))
            labels.append(f'L0 N{j}')
        # for bias neuron in input layer
        bias_x = 0 - layer_x_offset / 2
        bias_y = 0
        # positions.append((bias_x, bias_y))
        # labels.append(f'L0 Bias')        
        
        # Hidden and output layers
        for i, layer in enumerate(self.layers):
            layer_num = i + 1
            output_size = layer.output_size
            for j in range(output_size):
                x = layer_num * layer_x_offset
                y = (j - output_size / 2) * neuron_y_offset
                positions.append((x, y))
                labels.append(f'L{layer_num} N{j}')
            # for bias neuron in hidden and output layers
            bias_x = layer_num * layer_x_offset - layer_x_offset / 2
            bias_y = 0
            # positions.append((bias_x, bias_y))
            # labels.append(f'L{layer_num} Bias')
        
        # Create edges between layers
        for l, layer in enumerate(self.layers):
            input_size = layer.input_size
            output_size = layer.output_size
            
            # Get the starting and ending indices for this layer's neurons
            start_idx = sum(l.input_size for l in self.layers[:l]) if l > 0 else 0
            end_idx = start_idx + input_size
            
            next_start_idx = sum(l.output_size for l in self.layers[:l+1])
            next_end_idx = next_start_idx + output_size
            
            # Create edges between all input and output neurons
            for i in range(input_size):
                for j in range(output_size):
                    start_pos = positions[start_idx + i]
                    end_pos = positions[next_start_idx + j]
                    edges.append((start_pos, end_pos))
                    
                    # Position label at 1/4th of the edge length
                    label_x = start_pos[0] + 0.25 * (end_pos[0] - start_pos[0])
                    label_y = start_pos[1] + 0.25 * (end_pos[1] - start_pos[1])
                    
                    # Only show gradients if they exist
                    if hasattr(self, 'weights_grad') and len(self.weights_grad) > l:
                        grad_text = f'\n∇w={self.weights_grad[l][i, j]:.2f}' if l < len(self.weights_grad) else ''
                    else:
                        grad_text = ''
                    
                    edge_labels.append((label_x, label_y, 
                                    f'w={layer.weights[j, i]:.2f}{grad_text}'))
        
        # Create edge traces
        edge_traces = []
        for edge in edges:
            edge_traces.append(
                go.Scatter(
                    x=[edge[0][0], edge[1][0]],
                    y=[edge[0][1], edge[1][1]],
                    mode='lines',
                    line=dict(width=1, color='gray')
                )
            )
        
        # Create edge label traces
        label_traces = []
        for label_x, label_y, text in edge_labels:
            label_traces.append(
                go.Scatter(
                    x=[label_x],
                    y=[label_y],
                    mode='text',
                    text=[text],
                    textposition='middle center',
                    textfont=dict(size=8)
                )
            )
        
        # Create node trace
        node_trace = go.Scatter(
            x=[p[0] for p in positions],
            y=[p[1] for p in positions],
            mode='markers+text',
            marker=dict(size=15, color='lightblue', line=dict(width=2, color='darkblue')),
            text=labels,
            textposition='top center',
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + label_traces + [node_trace])
        fig.update_layout(
            title='Neural Network Architecture',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        fig.show()

    def plot_weights(self, layers: list[int]):
        for layer_number in layers:
            data = self.layers[layer_number].weights.flatten()
            bias_data = self.layers[layer_number].biases.flatten()

            plt.figure(figsize=(8, 6))
            sns.histplot(data, bins=10, kde=True)
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of weights in the {layer_number}-th layer")
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.histplot(bias_data, bins=10, kde=True, color="red")
            plt.xlabel("Bias value")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of biases in the {layer_number}-th layer")
            plt.show()

    def plot_gradients(self, layers: list[int]):
        for layer_number in layers:

            data = self.weights_grad[layer_number].flatten()
            plt.figure(figsize=(8, 6))
            sns.histplot(data, bins=10, kde=True)
            plt.xlabel("Weight gradient")
            plt.ylabel("Frequency")
            plt.title(
                f"Distribution of weights gradients in the {layer_number}-th layer"
            )
            plt.show()

            data = self.biases_grad[layer_number].flatten()
            plt.figure(figsize=(8, 6))
            sns.histplot(data, bins=10, kde=True, color="red")
            plt.xlabel("Bias gradient")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of bias gradients in the {layer_number}-th layer")
            plt.show()

    def plot_loss_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.title("Loss Function Value over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.show()

        # val loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.val_losses, label="Validation Loss")
        plt.title("Loss Function Value over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def fit(self, X, Y, validation_ratio=0.2):
        np.random.seed(self.random_state)

        epoch_range = (
            tqdm(range(self.epochs), desc="Training...")
            if self.verbose
            else range(self.epochs)
        )

        for epoch in epoch_range:
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            split_idx = int(len(X) * (1 - validation_ratio))
            X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
            Y_train, Y_val = Y_shuffled[:split_idx], Y_shuffled[split_idx:]

            this_epoch_loss = []

            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i : i + self.batch_size]
                Y_batch = Y_train[i : i + self.batch_size]

                # Forward pass
                self.forward(X_batch)

                # Compute loss
                y_pred = self.layers[-1].output_value
                loss = self.loss_function.calculate(Y_batch, y_pred)

                this_epoch_loss.append(loss)

                # Backward pass
                iteration_weights_grad, iteration_biases_grad = self.backward(Y_batch)

                if i == 0:
                    self.weights_grad = iteration_weights_grad
                    self.biases_grad = iteration_biases_grad
                else:
                    self.weights_grad += iteration_weights_grad
                    self.biases_grad += iteration_biases_grad

            # Only take the sum loss of the epoch for plotting
            self.train_losses.append(np.average(this_epoch_loss))
            self.val_losses.append(
                np.average(self.loss_function.calculate(Y_val, self.forward(X_val)))
            )

            if self.verbose:
                print(
                    f"Epoch {epoch + 1} - Training Loss: {self.train_losses[-1]}, Validation Loss: {self.val_losses[-1]}"
                )

    def forward(self, X):
        a = X
        for layer in self.layers:
            layer.input_value = a
            z = np.dot(a, layer.weights) + layer.biases
            a = layer.activation.activate(z)
            layer.z = z
            layer.output_value = a
        return a

    def backward(self, Y):
        y_pred = self.layers[-1].output_value

        # Handle output layer gradient
        if isinstance(self.loss_function, Loss.CategoricalCrossEntropy) and isinstance(
            self.layers[-1].activation, Activation.Softmax
        ):
            delta = y_pred - Y
        else:
            loss_derivative = self.loss_function.derivative(Y, y_pred)
            activation_derivative = self.layers[-1].activation.derivative(
                self.layers[-1].z
            )
            delta = loss_derivative * activation_derivative

        # Clip gradients to prevent exploding gradients
        delta = np.clip(delta, -1, 1)

        iteration_weights_grad = []
        iteration_biases_grad = []

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = layer.input_value

            # Compute gradients
            grad_weights = np.dot(a_prev.T, delta)
            grad_weights = np.clip(grad_weights, -1, 1)

            grad_biases = np.sum(delta, axis=0, keepdims=True)
            grad_biases = np.clip(grad_biases, -1, 1)

            # Update parameters
            layer.old_weights = layer.weights.copy()
            iteration_weights_grad.insert(0, grad_weights.copy())
            layer.weights -= self.learning_rate * grad_weights
            layer.old_biases = layer.biases.copy()
            iteration_biases_grad.insert(0, grad_biases.copy())
            layer.biases -= self.learning_rate * grad_biases

            # Propagate delta to previous layer
            if i > 0:
                delta = np.dot(delta, layer.old_weights.T)
                prev_activation_derivative = self.layers[i - 1].activation.derivative(
                    self.layers[i - 1].z
                )
                delta *= prev_activation_derivative
                delta = np.clip(delta, -1, 1)

        return iteration_weights_grad, iteration_biases_grad

    def predict(self, X):
        return self.forward(X)

    def set_weights(self, weights: list[np.ndarray]):
        assert len(weights) == len(
            self.layers
        ), "Number of weights should match number of layers"
        for i, weight in enumerate(weights):
            self.layers[i].weights = weight

    def set_biases(self, biases: list[np.ndarray]):
        assert len(biases) == len(
            self.layers
        ), "Number of biases should match number of layers"
        for i, bias in enumerate(biases):
            self.layers[i].biases = np.array([bias])

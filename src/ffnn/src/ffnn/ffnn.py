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
        GAP_MULTIPLIER = 200  # Horizontal spacing between layers
        NEURON_SPACING = 50  # Vertical spacing between neurons
        BIAS_OFFSET = -50  # Horizontal offset for bias neurons

        # Layer names
        layer_names = ["Input Layer"]
        for i in range(1, len(self.layers)):
            layer_names.append(f"Hidden Layer {i}")
        layer_names.append("Output Layer")

        # Initialize data structures
        nodes = []
        edges = []
        node_id_map = {}  # Maps (layer_idx, neuron_idx) to node ID
        node_idx = 0

        # Find maximum layer size for scaling
        max_layer_size = max(
            [layer.input_size for layer in self.layers] + [self.layers[-1].output_size]
        )

        # Create nodes for all neurons
        # Input layer (including bias)
        layer_idx = 0
        input_size = self.layers[0].input_size

        # Regular input neurons
        for neuron_idx in range(input_size):
            x = layer_idx * GAP_MULTIPLIER
            y = (neuron_idx - input_size / 2) * NEURON_SPACING
            nodes.append(
                {
                    "id": node_idx,
                    "label": f"L0 N{neuron_idx}",
                    "x": x,
                    "y": y,
                    "size": 15,
                    "color": "lightblue",
                    "layer": "Input Layer",
                    "type": "regular",
                }
            )
            node_id_map[(layer_idx, neuron_idx)] = node_idx
            node_idx += 1

        # Input layer bias neuron
        nodes.append(
            {
                "id": node_idx,
                "label": "L0 Bias",
                "x": layer_idx * GAP_MULTIPLIER + BIAS_OFFSET,
                "y": (input_size / 2 + 0.5) * NEURON_SPACING,
                "size": 15,
                "color": "pink",
                "layer": "Input Layer",
                "type": "bias",
            }
        )
        input_bias_id = node_idx
        node_idx += 1

        # Hidden and output layers
        for layer_idx, layer in enumerate(self.layers, start=1):
            output_size = layer.output_size
            is_output_layer = layer_idx == len(self.layers)

            # Regular neurons
            for neuron_idx in range(output_size):
                x = layer_idx * GAP_MULTIPLIER
                y = (neuron_idx - output_size / 2) * NEURON_SPACING
                nodes.append(
                    {
                        "id": node_idx,
                        "label": f"L{layer_idx} N{neuron_idx}",
                        "x": x,
                        "y": y,
                        "size": 15,
                        "color": "lightyellow" if not is_output_layer else "lightgreen",
                        "layer": layer_names[layer_idx],
                        "type": "regular",
                    }
                )
                node_id_map[(layer_idx, neuron_idx)] = node_idx
                node_idx += 1

            # Bias neuron (except for output layer)
            if not is_output_layer:
                nodes.append(
                    {
                        "id": node_idx,
                        "label": f"L{layer_idx} Bias",
                        "x": layer_idx * GAP_MULTIPLIER + BIAS_OFFSET,
                        "y": (output_size / 2 + 0.5) * NEURON_SPACING,
                        "size": 15,
                        "color": "pink",
                        "layer": layer_names[layer_idx],
                        "type": "bias",
                    }
                )
                node_id_map[(layer_idx, "bias")] = node_idx
                node_idx += 1

        # Create edges between layers
        for l, layer in enumerate(self.layers):
            input_size = layer.input_size
            output_size = layer.output_size

            # Get node indices for this layer
            if l == 0:
                # Input layer nodes
                input_nodes = [node_id_map[(0, i)] for i in range(input_size)]
                input_bias = input_bias_id
            else:
                # Previous layer's nodes
                input_nodes = [node_id_map[(l, i)] for i in range(input_size)]
                input_bias = node_id_map.get((l, "bias"), None)

            # Next layer's nodes
            output_nodes = [node_id_map[(l + 1, i)] for i in range(output_size)]

            # Regular connections
            for i, input_node in enumerate(input_nodes):
                for j, output_node in enumerate(output_nodes):
                    weight = layer.weights[j, i]
                    gradient = (
                        self.weights_grad[l][j, i]
                        if hasattr(self, "weights_grad") and l < len(self.weights_grad)
                        else 0
                    )

                    # Edge width based on weight magnitude
                    width = min(max(0.5, abs(weight) * 3), 5)
                    color = (
                        "rgba(123, 165, 209, 0.8)"
                        if weight >= 0
                        else "rgba(217, 123, 106, 0.8)"
                    )

                    edges.append(
                        {
                            "source": input_node,
                            "target": output_node,
                            "weight": weight,
                            "gradient": gradient,
                            "width": width,
                            "color": color,
                            "type": "regular",
                        }
                    )

            # Bias connections
            if input_bias is not None:
                for j, output_node in enumerate(output_nodes):
                    bias_value = layer.biases[0, j] if layer.biases.size > j else 0
                    edges.append(
                        {
                            "source": input_bias,
                            "target": output_node,
                            "weight": bias_value,
                            "gradient": (
                                self.biases_grad[l][0, j]
                                if hasattr(self, "biases_grad")
                                and l < len(self.biases_grad)
                                else 0
                            ),
                            "width": 1,
                            "color": "rgba(200, 100, 200, 0.8)",
                            "type": "bias",
                        }
                    )

        # Create node trace
        node_x = [node["x"] for node in nodes]
        node_y = [node["y"] for node in nodes]
        node_text = [f"{node['label']}<br>Layer: {node['layer']}" for node in nodes]
        node_colors = [node["color"] for node in nodes]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[node["label"] for node in nodes],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(color=node_colors, size=25, line=dict(width=2, color="black")),
        )

        # Create edge traces
        edge_traces = []
        for edge in edges:
            source_node = nodes[edge["source"]]
            target_node = nodes[edge["target"]]

            # Create multiple points along the edge for better hovering
            num_points = 20
            edge_x = []
            edge_y = []
            for i in range(num_points):
                ratio = i / (num_points - 1)
                edge_x.append(source_node["x"] * (1 - ratio) + target_node["x"] * ratio)
                edge_y.append(source_node["y"] * (1 - ratio) + target_node["y"] * ratio)

            # Format weight and gradient
            weight_display = f"{edge['weight']:.4f}"
            gradient_display = f"{edge['gradient']:.6f}"

            # Hover text
            hover_text = (
                f"<b>Connection:</b> {nodes[edge['source']]['label']} → {nodes[edge['target']]['label']}<br>"
                f"<b>Type:</b> {'Bias' if edge['type'] == 'bias' else 'Weight'}<br>"
                f"<b>Value:</b> {weight_display}<br>"
                f"<b>Gradient:</b> {gradient_display}"
            )

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=edge["width"], color=edge["color"]),
                mode="lines",
                hoverinfo="text",
                hovertemplate=hover_text + "<extra></extra>",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    bordercolor="black",
                ),
                opacity=0.7,
            )
            edge_traces.append(edge_trace)

        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="Neural Network Architecture Visualization",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-100, len(self.layers) * GAP_MULTIPLIER + 100],
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1,
                ),
                width=1200,
                height=800,
                plot_bgcolor="rgba(240, 240, 240, 0.2)",
            ),
        )

        # Add layer labels
        for i, name in enumerate(layer_names):
            fig.add_annotation(
                x=i * GAP_MULTIPLIER,
                y=(max_layer_size / 2 + 1) * NEURON_SPACING,
                text=name,
                showarrow=False,
                font=dict(size=16, color="black"),
            )

        # Add legend
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text="<b>Legend:</b><br>"
            "• Blue edges: Positive weights<br>"
            "• Red edges: Negative weights<br>"
            "• Purple edges: Bias connections<br>"
            "• Thickness indicates weight magnitude",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            align="left",
        )

        fig.show(
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "neural_network_visualization",
                    "height": 800,
                    "width": 1200,
                    "scale": 2,
                },
            }
        )

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

        iteration_weights_grad = []
        iteration_biases_grad = []

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = layer.input_value

            # Compute gradients
            grad_weights = np.dot(a_prev.T, delta)

            grad_biases = np.sum(delta, axis=0, keepdims=True)

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

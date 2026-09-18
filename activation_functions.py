"""Compare common activation functions on the same neuron outputs."""

import torch
from torch import nn


# Values a neuron might produce before applying an activation function.
raw_outputs = torch.linspace(-3, 3, steps=7)

activations = {
    "ReLU": nn.ReLU(),
    "Leaky ReLU": nn.LeakyReLU(negative_slope=0.01),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "GELU": nn.GELU(),
}

activated_outputs = {
    name: activation(raw_outputs) for name, activation in activations.items()
}

print("Comparison of activation functions")
names = list(activations)
print(f"{'raw input':>9} | " + " | ".join(f"{name:>10}" for name in names))
print("-" * (12 + 13 * len(names)))

for index, raw_output in enumerate(raw_outputs):
    values = " | ".join(
        f"{activated_outputs[name][index].item():10.3f}" for name in names
    )
    print(f"{raw_output.item():9.1f} | {values}")

print("\nReLU removes negative values and leaves positive values unchanged.")
print("Leaky ReLU keeps a small negative value instead of changing it to zero.")
print("Sigmoid maps every value to the range 0 to 1.")
print("Tanh maps every value to the range -1 to 1.")
print("GELU is a smooth activation function often used in Transformer models.")

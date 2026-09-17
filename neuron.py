import torch
from torch import nn

torch.manual_seed(42)

distance = torch.tensor([[5.0], [6.0]])  # shape: (examples, input_features)
minutes = torch.tensor([[22.2], [25.6]])

# One input -> one output: predicted_minutes = weight * distance + bias
neuron1 = nn.Linear(in_features=1, out_features=1)
neuron2 = nn.Sequential(nn.Linear(in_features=1, out_features=1))

loss_fn = nn.MSELoss()
models = {"neuron1": neuron1, "neuron2": neuron2}
optimizers = {
    name: torch.optim.SGD(model.parameters(), lr=0.01) for name, model in models.items()
}

# Training: repeatedly reduce each neuron's prediction error.
for epoch in range(1_000):
    for name, neuron in models.items():
        predictions = neuron(distance)
        loss = loss_fn(predictions, minutes)

        optimizers[name].zero_grad()
        loss.backward()  # calculate how the weight and bias should change
        optimizers[name].step()  # update the weight and bias

# Compare both implementations on a 7-mile delivery.
with torch.inference_mode():
    seven_miles = torch.tensor([[7.0]])
    predictions_at_seven_miles = {
        name: neuron(seven_miles).item() for name, neuron in models.items()
    }

print("Model comparison for a 7-mile delivery")
for name, neuron in models.items():
    # neuron2 is wrapped in Sequential, so its Linear layer is at index 0.
    linear_layer = neuron if isinstance(neuron, nn.Linear) else neuron[0]

    print(f"\n{name}")
    print(f"weight: {linear_layer.weight.item():.2f}")
    print(f"bias: {linear_layer.bias.item():.2f}")
    print(f"Predicted delivery time: {predictions_at_seven_miles[name]:.2f} minutes")

import torch
from torch import nn

torch.manual_seed(42)

distance = torch.tensor([[5.0], [6.0]])  # shape: (examples, input_features)
minutes = torch.tensor([[22.2], [25.6]])

# One input -> one output: predicted_minutes = weight * distance + bias
neuron = nn.Linear(in_features=1, out_features=1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(neuron.parameters(), lr=0.01)

# Training: repeatedly reduce prediction error
for epoch in range(1_000):
    predictions = neuron(distance)
    loss = loss_fn(predictions, minutes)

    optimizer.zero_grad()
    loss.backward()  # calculate how w and b should change
    optimizer.step()  # update w and b

# Predict the time for a 7-mile delivery
with torch.inference_mode():
    seven_miles = torch.tensor([[7.0]])
    predicted_minutes = neuron(seven_miles)

print(f"weight: {neuron.weight.item():.2f}")
print(f"bias: {neuron.bias.item():.2f}")
print(f"Predicted delivery time: {predicted_minutes.item():.2f} minutes")

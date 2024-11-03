import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Train the model
def startLinearRegressionModel(train_loader, test_loader, epochs = 100, stochastic_gradient_descent = 0.01 ):

    # Initialize the model
    model = LinearRegressionModel(input_dim=train_loader.dataset.tensors[0].shape[1])
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = torch.optim.SGD(model.parameters(), lr=stochastic_gradient_descent)  # Stochastic Gradient Descent

    # Train the model
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate the model
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        predictions = []
        targets = []

        for inputs, target in test_loader:
            output = model(inputs)
            predictions.extend(output.squeeze().numpy())
            targets.extend(target.numpy())

        # Convert to numpy arrays for metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        print(f'MSE: {mse}, R^2: {r2}')
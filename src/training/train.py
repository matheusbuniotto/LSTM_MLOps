import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model,
    train_dataset,
    test_dataset,
    num_epochs=50,
    batch_size=64,
    learning_rate=0.001,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # <-- ADD THIS LINE
    model = model.to(device)  # <-- MOVE MODEL TO DEVICE

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = (
                inputs.to(device),
                targets.to(device),
            )  # <-- USE device here
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # <-- Add this
    model = model.to(device)  # <-- Move model to device

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)  # <-- Use device, not model.device
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())
    return predictions, actuals

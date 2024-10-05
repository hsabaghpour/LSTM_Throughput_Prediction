# main.py
import torch
from torch.utils.data import DataLoader
from models.attention_lstm import AttentionLSTM
from utils.dataset import ThroughputDataset
from utils.data_generation import generate_data
from utils.plot_results import plot_results
from sklearn.metrics import mean_squared_error

# Sliding Window Preparation
def prepare_data(data, window_size, target_size, split_ratio=0.7):
    dataset = ThroughputDataset(data, window_size, target_size)
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_set, test_set

# Training function
def train(model, train_loader, optimizer, loss_fn, epochs=50):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

# Evaluation function
def evaluate(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            predictions.append(preds.numpy())
            actuals.append(y_batch.numpy())
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        return rmse, actuals, predictions

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    input_size = 1
    hidden_size = 64
    output_size = 1
    window_size = 10
    target_size = 1
    batch_size = 16
    learning_rate = 0.001
    epochs = 50

    # Generate and prepare data
    throughput_data = generate_data()
    train_set, test_set = prepare_data(throughput_data, window_size, target_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model setup
    model = AttentionLSTM(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Train and evaluate the model
    train(model, train_loader, optimizer, loss_fn, epochs=epochs)
    rmse, actual, predicted = evaluate(model, test_loader, loss_fn)
    print(f"Test RMSE: {rmse}")

    # Plot results
    plot_results(actual, predicted)

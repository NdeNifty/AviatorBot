import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data import results

# Sample data (replace this with your actual data)
data = results  # Example data, replace with your actual dataset

# Convert the data to a PyTorch tensor
data_tensor = torch.FloatTensor(data)

# Create sequences of 30 numbers
seq_length = 5
sequences = [data_tensor[i:i + seq_length] for i in range(len(data_tensor) - seq_length)]
targets = data_tensor[seq_length:]

# Prepare the dataset and dataloader
# Ensure that sequences and targets have the same size
assert len(sequences) == len(targets), "Size mismatch between sequences and targets"
dataset = TensorDataset(torch.stack(sequences), targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Instantiate the model
model = LSTMModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ...

# Training loop
num_epochs = 25  # Increase the number of epochs
for epoch in range(num_epochs):
    for seqs, targets in loader:
        # Reshape sequences for LSTM input
        seqs = seqs.view(-1, seq_length, 1)

        # Forward pass
        outputs = model(seqs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# ...


# Predict the next number in the sequence
# Here, we use the last sequence from the data for prediction
last_seq = data_tensor[-seq_length:].view(1, seq_length, 1)  # Ensure correct dimension for a single sequence
predicted_number = model(last_seq).item()
print(f'Predicted next number: {predicted_number}')

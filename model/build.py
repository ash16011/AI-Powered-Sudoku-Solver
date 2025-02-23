# File: model/build.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, Data
from model import SudokuGNN
import numpy as np
from datetime import datetime

# Directory for logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "build.log")

# Logging function
def log(message):
    """Logs messages to both console and file."""
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Training on device: {device}")

# Load preprocessed data
data_path = "data/processed/processed_sudoku_data_gnn.npz"
data = np.load(data_path)

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
Y_train = torch.tensor(data["Y_train"], dtype=torch.long)
X_val = torch.tensor(data["X_val"], dtype=torch.float32)
Y_val = torch.tensor(data["Y_val"], dtype=torch.long)
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
Y_test = torch.tensor(data["Y_test"], dtype=torch.long)
edge_index = torch.tensor(data["edge_index"], dtype=torch.long).to(device)

# Data Loaders
batch_size = 32  # Optimized for NVIDIA 3060 Laptop GPU
train_loader = DataLoader([Data(x=X_train[i], y=Y_train[i], edge_index=edge_index) for i in range(len(X_train))], 
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader([Data(x=X_val[i], y=Y_val[i], edge_index=edge_index) for i in range(len(X_val))], 
                        batch_size=batch_size)
test_loader = DataLoader([Data(x=X_test[i], y=Y_test[i], edge_index=edge_index) for i in range(len(X_test))], 
                         batch_size=batch_size)

# Initialize Model
model = SudokuGNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Early Stopping Parameters
best_val_loss = float('inf')
patience = 20
patience_counter = 0

# Mixed Precision Setup
scaler = torch.amp.GradScaler()

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(batch.x, batch.edge_index)
            loss = criterion(outputs.view(-1, 9), batch.y.view(-1))

        # Mixed Precision Backpropagation
        scaler.scale(loss).backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    # Calculate Training Loss
    avg_train_loss = total_loss / len(train_loader)
    log(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            loss = criterion(outputs.view(-1, 9), batch.y.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    log(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Learning Rate Scheduler Step
    scheduler.step(avg_val_loss)

    # Early Stopping and Model Saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        model_save_path = "model/trained_sudoku_gnn.pth"
        torch.save(model.state_dict(), model_save_path)
        log(f"Validation Loss improved. Model saved at {model_save_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            log("Early stopping triggered.")
            break

# Testing Phase
log("Starting testing phase...")
model.load_state_dict(torch.load("model/trained_sudoku_gnn.pth"))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index)
        
        # Flatten Predictions and Targets for Accuracy Calculation
        preds = torch.argmax(outputs.view(-1, 9), dim=1)
        targets = batch.y.view(-1)
        
        correct += (preds == targets).sum().item()
        total += targets.numel()

test_accuracy = (correct / total) * 100
log(f"Test Accuracy: {test_accuracy:.2f}%")

# Final Model Save
final_model_path = "model/trained_sudoku_gnn_final.pth"
torch.save(model.state_dict(), final_model_path)
log(f"Final trained model saved as '{final_model_path}'")

log("Training and evaluation completed successfully!")

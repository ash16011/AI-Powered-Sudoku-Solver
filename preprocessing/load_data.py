# File: preprocessing/load_data.py

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define folder paths
RAW_DATA_PATH = "data/raw/sudoku-3m.csv"
PROCESSED_DATA_PATH = "data/processed"
LOGS_FOLDER = "logs"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Function to convert a Sudoku string to a 9x9 grid
def string_to_grid(sudoku_str):
    """Convert a Sudoku string to a 9x9 NumPy array."""
    sudoku_str = sudoku_str.replace(".", "0")  # Convert '.' to '0' for empty cells
    grid = np.array([int(char) for char in sudoku_str], dtype=np.int32).reshape(9, 9)
    return grid

# Function to one-hot encode the input grid
def one_hot_encode(grid):
    """
    One-hot encode a 9x9 Sudoku grid.
    - Input: (9, 9) grid with digits 0-9
    - Output: (81, 9) one-hot encoded tensor (81 nodes, 9 features per node)
    """
    one_hot = np.zeros((81, 9), dtype=np.float32)
    for row in range(9):
        for col in range(9):
            digit = grid[row, col]
            if digit > 0:
                one_hot[row * 9 + col, digit - 1] = 1  # One-hot encode digits 1-9 as indices 0-8
    return one_hot

# Function to build graph structure
def build_sudoku_graph():
    """Builds the graph structure for a 9x9 Sudoku grid."""
    edge_index = []

    # Row connections
    for row in range(9):
        for col in range(9):
            for k in range(9):
                if k != col:
                    edge_index.append([row * 9 + col, row * 9 + k])  # Same row

    # Column connections
    for col in range(9):
        for row in range(9):
            for k in range(9):
                if k != row:
                    edge_index.append([row * 9 + col, k * 9 + col])  # Same column

    # Subgrid connections
    for box_row in range(3):
        for box_col in range(3):
            nodes = []
            for i in range(3):
                for j in range(3):
                    nodes.append((box_row * 3 + i) * 9 + (box_col * 3 + j))
            for u in nodes:
                for v in nodes:
                    if u != v:
                        edge_index.append([u, v])  # Same 3x3 box

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)  # Make edges undirected
    return edge_index

# Load dataset
print("Loading raw Sudoku dataset...")
df = pd.read_csv(RAW_DATA_PATH)
df = df[['puzzle', 'solution']]
df = df.sample(frac=0.25, random_state=42).reset_index(drop=True)  # Reduce dataset size to 25%

# Convert strings to 9x9 grids
puzzles = [string_to_grid(puzzle) for puzzle in df['puzzle']]
solutions = [string_to_grid(solution) for solution in df['solution']]

# One-hot encode the puzzles for input
X = np.array([one_hot_encode(puzzle) for puzzle in puzzles])  # Shape: (num_samples, 81, 9)
Y = np.array([solution.flatten() - 1 for solution in solutions])  # Shape: (num_samples, 81)

# Train-Validation-Test Split
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

# Build Graph Structure
edge_index = build_sudoku_graph()

# Save processed data as a compressed NPZ file
processed_file = os.path.join(PROCESSED_DATA_PATH, "processed_sudoku_data_gnn.npz")
np.savez_compressed(processed_file, 
                    X_train=X_train, Y_train=Y_train,
                    X_val=X_val, Y_val=Y_val,
                    X_test=X_test, Y_test=Y_test,
                    edge_index=edge_index.numpy())

print(f"Processed data saved to {processed_file}")

# Save preprocessing logs with timestamp
log_file = os.path.join(LOGS_FOLDER, "preprocessing.log")
with open(log_file, "a") as log:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.write(f"[{timestamp}] Dataset loaded from: {RAW_DATA_PATH}\n")
    log.write(f"[{timestamp}] Reduced dataset size: {len(df)} puzzles\n")
    log.write(f"[{timestamp}] Processed data saved to: {processed_file}\n")

# Display first few rows to verify processing
print(f"\nFirst puzzle (one-hot encoded):\n{X_train[0]}")
print(f"\nFirst solution (class labels):\n{Y_train[0]}")

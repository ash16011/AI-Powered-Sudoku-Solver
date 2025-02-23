# File: solve.py
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch_geometric.data import Data
from model.model_gnn import SudokuGNN
import torch.nn.functional as F
from copy import deepcopy
import time  # Added for time measurement

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = SudokuGNN().to(device)
# Get the absolute path of the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_sudoku_gnn_final.pth')
model_path = os.path.abspath(model_path)
# Load the model
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully.")

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
    return edge_index

# Function to check if placing a number is valid
def is_valid(puzzle, row, col, num):
    """Checks if it's valid to place num at puzzle[row][col]."""
    # Check row
    if num in puzzle[row]:
        return False

    # Check column
    if num in puzzle[:, col]:
        return False

    # Check subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in puzzle[start_row:start_row + 3, start_col:start_col + 3]:
        return False

    return True

# DFS Backtracking Algorithm
def dfs_backtrack(puzzle, empty_cells, index=0):
    """
    Solves the Sudoku puzzle using DFS backtracking.
    - empty_cells: List of tuples with (row, col, probabilities)
    - index: Index of the current cell to fill
    """
    if index == len(empty_cells):
        return True  # All cells are filled successfully

    row, col, probs = empty_cells[index]
    sorted_digits = np.argsort(-probs) + 1  # Sort digits by descending probability

    for num in sorted_digits:
        if is_valid(puzzle, row, col, num):
            puzzle[row, col] = num
            if dfs_backtrack(puzzle, empty_cells, index + 1):
                return True
            puzzle[row, col] = 0  # Reset and backtrack

    return False  # No valid number found, trigger backtracking

# Function to solve a Sudoku puzzle
def solve_sudoku(puzzle):
    """
    Solves a Sudoku puzzle using the trained GNN model and DFS backtracking.
    - Input: (9, 9) NumPy array with digits 0-9 (0 = empty cell)
    - Output: (9, 9) NumPy array with solved puzzle
    """
    # Start Time Measurement
    start_time = time.time()

    # One-hot encode the input puzzle
    node_features = one_hot_encode(puzzle)
    node_features = torch.tensor(node_features, dtype=torch.float32).to(device)

    # Build graph structure
    edge_index = build_sudoku_graph().to(device)

    # Create Data object for PyTorch Geometric
    data = Data(x=node_features, edge_index=edge_index)

    # Perform forward pass
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        output = output.view(-1, 9)  # Flatten to (81, 9)
        probs = F.softmax(output, dim=1).cpu().numpy()

    # Decode predictions and identify empty cells
    solved_puzzle = deepcopy(puzzle)
    empty_cells = []

    for row in range(9):
        for col in range(9):
            if puzzle[row, col] == 0:
                idx = row * 9 + col
                empty_cells.append((row, col, probs[idx]))

    # DFS Backtracking to correct the puzzle
    if dfs_backtrack(solved_puzzle, empty_cells):
        # End Time Measurement
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"\nSolved in {time_taken:.4f} seconds using GNN + Backtracking.")
        return solved_puzzle, time_taken
    else:
        print("No solution found.")
        return puzzle, None  # Return the initial puzzle if no solution

# Input: More Complex Sudoku Puzzle (0 = Empty Cell)
example_puzzle = np.array([
    [6, 0, 0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 3, 5, 0, 0, 0],
    [0, 4, 5, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 6],
    [0, 1, 0, 0, 0, 0, 0, 5, 0],
    [9, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 2, 4, 0],
    [0, 0, 0, 4, 7, 0, 0, 0, 0],
    [1, 7, 0, 0, 0, 0, 0, 0, 0]
])

# Solve the puzzle
solved_puzzle, time_taken = solve_sudoku(example_puzzle)

print("\nInput Puzzle:")
print(example_puzzle)
print("\nSolved Puzzle:")
print(solved_puzzle)

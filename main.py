from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
from torch_geometric.data import Data
from model.model_gnn import SudokuGNN
import torch.nn.functional as F
from copy import deepcopy

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuGNN().to(device)
model.load_state_dict(torch.load("model/trained_sudoku_gnn_final.pth"))
model.eval()
print("Model loaded successfully.")

# Function to one-hot encode the input grid
def one_hot_encode(grid):
    one_hot = np.zeros((81, 9), dtype=np.float32)
    for row in range(9):
        for col in range(9):
            digit = grid[row, col]
            if digit > 0:
                one_hot[row * 9 + col, digit - 1] = 1  # One-hot encode digits 1-9 as indices 0-8
    return one_hot

# Function to build graph structure
def build_sudoku_graph():
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
    if num in puzzle[row]:
        return False
    if num in puzzle[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in puzzle[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    return True

# DFS Backtracking Algorithm
def dfs_backtrack(puzzle, empty_cells, index=0):
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
    node_features = one_hot_encode(puzzle)
    node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_index = build_sudoku_graph().to(device)
    data = Data(x=node_features, edge_index=edge_index)

    with torch.no_grad():
        output = model(data.x, data.edge_index)
        output = output.view(-1, 9)
        probs = F.softmax(output, dim=1).cpu().numpy()

    solved_puzzle = deepcopy(puzzle)
    empty_cells = []

    for row in range(9):
        for col in range(9):
            if puzzle[row, col] == 0:
                idx = row * 9 + col
                empty_cells.append((row, col, probs[idx]))

    if dfs_backtrack(solved_puzzle, empty_cells):
        return solved_puzzle
    else:
        print("No solution found.")
        return puzzle

# ✅ New Function: Parse Puzzle Input
def parse_puzzle_input(puzzle_input):
    numbers = [int(x) for x in puzzle_input.replace("\n", ",").split(",") if x.strip().isdigit()]

    # Ensure there are exactly 81 numbers
    if len(numbers) != 81:
        raise ValueError("Invalid input. Must contain exactly 81 numbers.")

    # Convert the flat list into a 9x9 grid
    grid = [numbers[i:i+9] for i in range(0, len(numbers), 9)]
    return grid

# ✅ Updated Home Route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "puzzle": None, "solved_puzzle": None})

# ✅ Updated Solve Route
@app.post("/solve", response_class=HTMLResponse)
async def solve(request: Request):
    form_data = await request.form()
    puzzle_input = form_data.get("puzzle")

    try:
        grid = parse_puzzle_input(puzzle_input)
    except ValueError as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

    solved_puzzle = solve_sudoku(np.array(grid))

    return templates.TemplateResponse("index.html", {
        "request": request,
        "puzzle": grid,
        "solved_puzzle": solved_puzzle
    })

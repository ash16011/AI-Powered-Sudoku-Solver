# File: solve_backtracking.py

import numpy as np
import time

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
def dfs_backtrack(puzzle):
    """
    Solves the Sudoku puzzle using pure DFS backtracking.
    - Input: (9, 9) NumPy array with digits 0-9 (0 = empty cell)
    - Output: True if a solution is found, False otherwise
    """
    # Find the first empty cell
    for row in range(9):
        for col in range(9):
            if puzzle[row, col] == 0:
                # Try placing numbers 1 to 9
                for num in range(1, 10):
                    if is_valid(puzzle, row, col, num):
                        puzzle[row, col] = num
                        if dfs_backtrack(puzzle):
                            return True
                        puzzle[row, col] = 0  # Reset and backtrack
                return False  # No valid number found, trigger backtracking
    return True  # All cells are filled successfully

# Function to solve a Sudoku puzzle using pure backtracking
def solve_sudoku(puzzle):
    """
    Solves a Sudoku puzzle using pure DFS backtracking.
    - Input: (9, 9) NumPy array with digits 0-9 (0 = empty cell)
    - Output: (9, 9) NumPy array with solved puzzle
    """
    solved_puzzle = puzzle.copy()

    # Start Time Measurement
    start_time = time.time()

    if dfs_backtrack(solved_puzzle):
        # End Time Measurement
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"\nSolved in {time_taken:.4f} seconds using pure backtracking.")
        return solved_puzzle, time_taken
    else:
        print("\nNo solution found.")
        return puzzle, None  # Return the initial puzzle if no solution

# Input: Complex Sudoku Puzzle (0 = Empty Cell)
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

print("\nInput Puzzle:")
print(example_puzzle)

# Solve the puzzle using pure backtracking
solved_puzzle, time_taken = solve_sudoku(example_puzzle)

print("\nSolved Puzzle:")
print(solved_puzzle)

# Log time taken
if time_taken:
    print(f"\nTime Taken: {time_taken:.4f} seconds")
else:
    print("\nNo solution found.")

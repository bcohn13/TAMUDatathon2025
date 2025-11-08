import math
from collections import deque
import case_closed_game
import torch  # PyTorch import for CPU-only tensor operations

# Convert GameBoard grid to a torch tensor for efficient CPU use
def board_to_tensor(board: case_closed_game.GameBoard) -> torch.Tensor:
    grid = board.grid
    return torch.tensor(grid, dtype=torch.int8, device='cpu')

# BFS to count reachable spaces from a starting position on the board using PyTorch tensor
def bfs_reachable(board: case_closed_game.GameBoard, start: tuple[int, int]) -> int:
    board_tensor = board_to_tensor(board)
    visited = set()
    q = deque([start])
    visited.add(start)
    count = 0

    while q:
        pos = q.popleft()
        count += 1
        for dr, dc in [Direction.UP.value, Direction.DOWN.value, Direction.RIGHT.value, Direction.LEFT.value]:
            nr, nc = board._torus_check((pos[0] + dr, pos[1] + dc))
            if board_tensor[nc, nr] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return count

# Heuristic evaluating difference in reachable spaces
def evaluate_state(board: case_closed_game.GameBoard, my_pos: tuple[int, int], opp_pos: tuple[int, int]) -> int:
    return bfs_reachable(board, my_pos) - bfs_reachable(board, opp_pos)

def is_terminal(board: case_closed_game.GameBoard, my_pos: tuple[int, int], opp_pos: tuple[int, int]) -> bool:
    board_tensor = board_to_tensor(board)
    if board_tensor[my_pos[1], my_pos[0]] == 1 or board_tensor[opp_pos[1], opp_pos[0]] == 1:
        return True
    if my_pos == opp_pos:
        return True
    return False

def get_valid_moves(board: case_closed_game.GameBoard, pos: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
    board_tensor = board_to_tensor(board)
    valid_moves = []
    for i, (dr, dc) in enumerate([Direction.UP.value, Direction.DOWN.value, Direction.RIGHT.value, Direction.LEFT.value]):
        new_pos = board._torus_check((pos[0] + dr, pos[1] + dc))
        if board_tensor[new_pos[1], new_pos[0]] == 0:
            valid_moves.append((i, new_pos))
    return valid_moves

def minimax(board: case_closed_game.GameBoard, my_pos: tuple[int, int], opp_pos: tuple[int, int], depth: int, maximizing_player: bool):
    if depth == 0 or is_terminal(board, my_pos, opp_pos):
        return evaluate_state(board, my_pos, opp_pos), None

    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        for action, new_pos in get_valid_moves(board, my_pos):
            new_board = copy_board_with_tensor(board, new_pos)  # Create new board with updated state
            eval_score, _ = minimax(new_board, new_pos, opp_pos, depth - 1, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = action
        return max_eval, best_move
    else:
        min_eval = math.inf
        best_move = None
        for action, new_pos in get_valid_moves(board, opp_pos):
            new_board = copy_board_with_tensor(board, new_pos)
            eval_score, _ = minimax(new_board, my_pos, new_pos, depth - 1, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = action
        return min_eval, best_move

# Helper to clone board and mark a new trail position efficiently using PyTorch tensors
def copy_board_with_tensor(board: case_closed_game.GameBoard, new_trail_pos: tuple[int, int]) -> case_closed_game.GameBoard:
    new_board = case_closed_game.GameBoard(board.height, board.width)
    # Convert current board to tensor and clone
    original_tensor = board_to_tensor(board)
    cloned_tensor = original_tensor.clone()
    # Mark new trail position
    x, y = new_trail_pos
    cloned_tensor[y, x] = 1

    # Update new_board.grid from cloned_tensor
    for row in range(board.height):
        for col in range(board.width):
            new_board.grid[row][col] = int(cloned_tensor[row, col].item())

    return new_board

# Usage remains same:
# game = case_closed_game.Game()
# current_board = game.board
# agent_pos = game.agent1.trail[-1]
# opponent_pos = game.agent2.trail[-1]
# score, best_action = minimax(current_board, agent_pos, opponent_pos, depth=3, maximizing_player=True)
# direction = [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT][best_action]


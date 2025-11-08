import math
from collections import deque
from enum import Direction
import case_closed_game
from copy import deepcopy

# BFS to count reachable spaces from a starting position on the board
def bfs_reachable(board: case_closed_game.GameBoard, start: tuple[int, int]) -> int:
    visited = set()
    q = deque([start])
    visited.add(start)
    count = 0

    while q:
        pos = q.popleft()
        count += 1
        for dr, dc in [Direction.UP.value, Direction.DOWN.value, Direction.RIGHT.value, Direction.LEFT.value]:
            nr, nc = board._torus_check((pos[0] + dr, pos[1] + dc))
            if board.get_cell_state((nr, nc)) == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return count

# Simple heuristic: difference in reachable spaces
def evaluate_state(board: case_closed_game.GameBoard, my_pos: tuple[int, int], opp_pos: tuple[int, int]) -> int:
    my_space = bfs_reachable(board, my_pos)
    opp_space = bfs_reachable(board, opp_pos)
    return my_space - opp_space

# Terminal test: if an agent moved into a trail or same position
def is_terminal(board: case_closed_game.GameBoard, my_pos: tuple[int, int], opp_pos: tuple[int, int]) -> bool:
    if board.get_cell_state(my_pos) == 1 or board.get_cell_state(opp_pos) == 1:
        return True
    if my_pos == opp_pos:
        return True
    return False

# Generate valid moves for an agent from their current position
def get_valid_moves(board: case_closed_game.GameBoard, pos: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
    valid_moves = []
    for i, (dr, dc) in enumerate([Direction.UP.value, Direction.DOWN.value, Direction.RIGHT.value, Direction.LEFT.value]):
        new_pos = board._torus_check((pos[0] + dr, pos[1] + dc))
        if board.get_cell_state(new_pos) == 0:
            valid_moves.append((i, new_pos))
    return valid_moves

# Minimax function alternating between maximizing and minimizing player
def minimax(board: case_closed_game.GameBoard, my_pos: tuple[int, int], opp_pos: tuple[int, int], depth: int, maximizing_player: bool):
    if depth == 0 or is_terminal(board, my_pos, opp_pos):
        score = evaluate_state(board, my_pos, opp_pos)
        return score, None

    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        for action, new_pos in get_valid_moves(board, my_pos):
            new_board = deepcopy(board)
            new_board.set_cell_state(new_pos, 1)  # mark trail
            eval_score, _ = minimax(new_board, new_pos, opp_pos, depth - 1, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = action
        return max_eval, best_move
    else:
        min_eval = math.inf
        best_move = None
        for action, new_pos in get_valid_moves(board, opp_pos):
            new_board = deepcopy(board)
            new_board.set_cell_state(new_pos, 1)
            eval_score, _ = minimax(new_board, my_pos, new_pos, depth - 1, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = action
        return min_eval, best_move

# Usage:
# game = Game()
# current_board = game.board
# agent_pos = game.agent1.trail[-1]
# opponent_pos = game.agent2.trail[-1]
# score, best_action = minimax(current_board, agent_pos, opponent_pos, depth=3, maximizing_player=True)
# direction = [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT][best_action]


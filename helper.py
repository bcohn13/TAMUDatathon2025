import torch
from collections import deque

def count_reachable_spaces(board, start_pos):
    """
    Args:
        board: (H x W) torch tensor, 0 for empty, 1 for wall
        start_pos: (row, col) tuple
    Returns:
        int: count of reachable empty cells
        set: all positions that are reachable
    """
    H, W = board.shape
    visited = set()
    q = deque()
    q.append(start_pos)
    visited.add(start_pos)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    spaces = 0

    while q:
        r, c = q.popleft()
        spaces += 1
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and 
                board[nr, nc].item() == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                q.append((nr, nc))
    return spaces, visited

# Example usage
board = torch.zeros((18, 20), dtype=torch.int64)
# Set some cells as wall/trail: board[4,7] = 1
my_start = (2, 3)
opp_start = (10, 15)
my_spaces, my_area = count_reachable_spaces(board, my_start)
opp_spaces, opp_area = count_reachable_spaces(board, opp_start)
print("My reachable spaces:", my_spaces)
print("Opponent reachable spaces:", opp_spaces)

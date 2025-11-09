import numpy as np
from case_closed_game import Game, Direction, GameResult

class CaseClosedEnv:
    def __init__(self):
        self.game = Game()
    
    def reset(self):
        """Reset the game and return initial state as 18x20 numpy array"""
        self.game.reset()
        return np.array(self.game.board.grid, dtype=np.float32)
    
    def step(self, action):
        """
        Take an action for agent1. 
        For training, agent2 can be a simple random agent.
        
        Returns:
            next_state: 18x20 numpy array
            reward: float
            done: bool
        """
        # Map numeric action to Direction
        action_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT
        }
        agent1_dir = action_map[action]
        
        # Use random move for agent2
        possible_moves = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        agent2_dir = np.random.choice(possible_moves)
        
        # Step the game
        result = self.game.step(agent1_dir, agent2_dir)
        
        # Reward shaping
        if result == GameResult.AGENT1_WIN:
            reward = 1.0
            done = True
        elif result == GameResult.AGENT2_WIN:
            reward = -1.0
            done = True
        elif result == GameResult.DRAW:
            reward = 0.0
            done = True
        else:
            reward = 0.1  # small reward for surviving
            done = False
        
        next_state = np.array(self.game.board.grid, dtype=np.float32)
        return next_state, reward, done
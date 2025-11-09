import numpy as np
from case_closed_game import Game, Direction, GameResult
from time import sleep

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


        agent1_dir = action
        
        # Use random move for agent2
        moves = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT] #Pool of moves
        possible_moves = []

        #Check if move is valid
        agent2_head = self.game.agent2.trail[-1]
        cur_dx, cur_dy = self.game.agent2.direction.value
        for move in moves:
            req_dx, req_dy = move.value
            #if (req_dx, req_dy) != (-cur_dx, -cur_dy) and self.game.board.grid[agent2_head[0]+move.value[0]][agent2_head[1]+move.value[1]] != 1: #needs to move such that it doesn't ram into itself or suicide 
            #pos = self.game.board._torus_check((agent2_head[0]+move.value[0],agent2_head[1]+move.value[1]))
            if (req_dx, req_dy) != (-cur_dx, -cur_dy): #I don't think im checkin th
                possible_moves.append(move) #add to list of potential moves if yes

        agent2_dir = np.random.choice(possible_moves)
        
        # Step the game
        result = self.game.step(agent1_dir, agent2_dir)
        #print(self.game.board)
        #print(agent1_dir)
        #sleep(2)
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
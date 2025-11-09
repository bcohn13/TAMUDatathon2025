import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import helper
import torch
from case_closed_game import Game, Direction, GameResult
import numpy as np



state_array = np.random.randint(0, 2, size=(18, 20)) #hard code some array
device = torch.device("cpu")
policy_net = helper.DQN().to(device)
move = helper.select_action(state_array, policy_net, epsilon=0.0, device=device)
print(move)
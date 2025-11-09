import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import helper
import torch
from case_closed_game import Game, Direction, GameResult
import numpy as np

import train_model

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"



game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)
    action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    device = torch.device("cpu")
    policy_net = helper.DQN().to(device)
# Load weights if you have a saved model
# policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))

    #CHANGE THIS TO FALSE WHEN SUBMIT FOR ACTUAL SUBMISSION !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    isTraining = True
    #CHANGE THIS TO FALSE WHEN SUBMIT FOR ACTUAL SUBMISSION !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    policy_net.eval()
    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
    grid_data = state['board']  # adjust key to your actual state structure
    state_array = np.array(grid_data, dtype=np.float32)  

    if (not isTraining):
        trained_policy = helper.DQN().to(device)
        trained_policy.load_state_dict(torch.load("policy_net.pth"))
        trained_policy.eval()
        move = helper.select_action(state_array, trained_policy, epsilon=0.0, device=device)
    elif (isTraining):
        #We will run through the step > reward > update buffer loop
        move = helper.select_action(state_array, policy_net, train_model.epsilon, device)
        x, y = state["agent1_trail"]
        normalized_x = x % 20
        normalized_y = y % 18
        newState = state_array[normalized_y+move.value[0]][normalized_x+move.value[1]]
        train_model.updateModel(move, state_array, newState, 10)

    if (move == Direction.DOWN):
        move = "DOWN"
    elif (move == Direction.UP):
        move = "UP"
    if (move == Direction.RIGHT):
        move = "RIGHT"
    elif (move == Direction.LEFT):
        move = "LEFT"
   
    # -----------------your code here-------------------
    # Simple example: always go RIGHT (replace this with your logic)
    # To use a boost: move = "RIGHT:BOOST"
    
    # Example: Use boost if available and it's late in the game
    # turn_count = state.get("turn_count", 0)
    # if boosts_remaining > 0 and turn_count > 50:
    #     move = "RIGHT:BOOST"
    # -----------------end code here--------------------
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)

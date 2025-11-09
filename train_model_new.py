import subprocess
import logging
from time import sleep
import sys
import json

python_exe = sys.executable
agent1 = subprocess.Popen([python_exe, "agent.py"])
agent2 = subprocess.Popen([python_exe,"sample_agent.py"])
sleep(3)
engine = subprocess.run([python_exe,"judge_engine.py"])

with open("rewards.json", "r") as f:
    reward = json.load(f)

print(reward["reward"])
# judging_engine.py should print or log reward information
# e.g. "REWARD: 1" or "AGENT_WIN"

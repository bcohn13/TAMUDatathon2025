import subprocess
import logging
from time import sleep
import sys

python_exe = sys.executable
subprocess.Popen([python_exe, "agent.py"])
subprocess.Popen([python_exe,"sample_agent.py"])
sleep(3)
results = subprocess.Popen([python_exe,"judge_engine.py"], capture_output=True,)

output = results.stdout

print(output)
# judging_engine.py should print or log reward information
# e.g. "REWARD: 1" or "AGENT_WIN"

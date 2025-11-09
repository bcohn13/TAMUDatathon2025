import torch
from helper import DQN, train_dqn  # import your DQN code
from training_env import CaseClosedEnv  # the wrapper we discussed

def main():
    env = CaseClosedEnv()  # create the environment
    train_dqn(env, num_episodes=200)  # train for 500 episodes (adjust as needed)

    # Optionally, save your trained model
    model = DQN()
    torch.save(model.state_dict(), "trained_model.pth")
    print("Training finished and model saved!")

if __name__ == "__main__":
    main()
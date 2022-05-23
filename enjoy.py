import numpy as np
import pickle
import torch
from docopt import docopt
from model import ActorCriticModel
from utils import create_env

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Instantiate environment
    env = create_env(config["env"])

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False
    episode_rewards = []
    memory_mask = torch.tril(torch.ones((env.max_episode_steps, env.max_episode_steps)))
    # Shift mask by one to account for the fact that for the first timestep the memory is empty
    memory_mask = torch.cat((torch.zeros((1, env.max_episode_steps)), memory_mask))[:-1]  
    memory = torch.zeros((1, env.max_episode_steps, config["episodic_memory"]["num_layers"], config["episodic_memory"]["layer_size"]), dtype=torch.float32)
    current_memory_item = torch.zeros((1, 1, config["episodic_memory"]["num_layers"], config["episodic_memory"]["layer_size"]), dtype=torch.float32)
    t = 0

    obs = env.reset()
    while not done:
        # Render environment
        env.render()
        # Forward model
        policy, value, current_memory_item = model(torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32), memory, memory_mask[t].unsqueeze(0))
        memory[:, t] = current_memory_item
        # Sample action
        action = policy.sample().cpu().numpy()
        # Step environemnt
        obs, reward, done, info = env.step(int(action))
        episode_rewards.append(reward)
        t += 1
    
    # after done, render last state
    env.render()

    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

    env.close()

if __name__ == "__main__":
    main()
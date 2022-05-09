import numpy as np
import torch
import string

from gym import spaces
from buffer import Buffer


def main():
    # Init buffer
    config = {
        "n_workers": 4,
        "worker_steps": 8,
        "n_mini_batch": 4,
        "episodic_memory":
        {
            "num_layers": 2,
            "layer_size": 4
        }
    }
    max_episode_length = 12
    obs_space = spaces.Box(low = 0, high = 1.0, shape = (1,), dtype = np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = Buffer(config, obs_space, max_episode_length, device)

    obs_0 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    obs_1 = torch.tensor([5, 6, 7, 8, 9, 10, 11, 0])
    obs_2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    obs_3 = torch.tensor([6, 0, 1, 2, 3, 4, 0, 1])
    buffer.obs = torch.unsqueeze(torch.stack((obs_0, obs_1, obs_2, obs_3)), dim=2)

    dones_0 = np.asarray([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    dones_1 = np.asarray([0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    dones_2 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    dones_3 = np.asarray([1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    buffer.dones = np.stack((dones_0, dones_1, dones_2, dones_3))
    
    # TODO Memories, IN Memory, Worker Indices
    memory_0 = torch.tensor([])
    memory_1 = torch.tensor([])
    memory_2 = torch.tensor([])
    memory_3 = torch.tensor([])
    in_0 = torch.tensor([])
    in_1 = torch.tensor([])
    in_2 = torch.tensor([])
    in_3 = torch.tensor([])
    buffer.timestep = torch.tensor([obs_0[0], obs_1[0], obs_2[0], obs_3[0]])

    print(string.ascii_lowercase)


if __name__ == "__main__":
    main()
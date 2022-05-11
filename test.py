import numpy as np
import torch

from gym import spaces
from buffer import Buffer


def main():
    # Set seed
    # torch.manual_seed(800)
    print("SEED")
    print(torch.seed())
    print("---------------")

    # Init buffer
    config = {
        "n_workers": 4,
        "worker_steps": 8,
        "n_mini_batch": 4,
        "episodic_memory":
        {
            "num_layers": 2,
            "layer_size": 1
        }
    }
    max_episode_length = 12
    obs_space = spaces.Box(low = 0, high = 1.0, shape = (1,), dtype = np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = Buffer(config, obs_space, max_episode_length, device)

    # Observation Data
    obs_0 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    obs_1 = torch.tensor([5, 6, 7, 8, 9, 10, 11, 0])
    obs_2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    obs_3 = torch.tensor([6, 0, 1, 2, 3, 4, 0, 1])
    buffer.obs = torch.unsqueeze(torch.stack((obs_0, obs_1, obs_2, obs_3)), dim=2)

    # Done Data
    dones_0 = np.asarray([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    dones_1 = np.asarray([0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    dones_2 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    dones_3 = np.asarray([1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    buffer.dones = np.stack((dones_0, dones_1, dones_2, dones_3))
    
    # Timesteps data
    steps_0 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    steps_1 = torch.tensor([5, 6, 7, 8, 9, 10, 11, 0])
    steps_2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    steps_3 = torch.tensor([6, 0, 1, 2, 3, 4, 0, 1])
    buffer.timesteps = torch.stack((steps_0, steps_1, steps_2, steps_3))
    
    # Episodic Memory Data
    memory_0 = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2], [0, 1, 2, 3, 4, 0, 1, 2]])
    memory_1 = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 0], [5, 6, 7, 8, 9, 10, 11, 0]])
    memory_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
    memory_3 = torch.tensor([[6, 0, 1, 2, 3, 4, 0, 1], [6, 0, 1, 2, 3, 4, 0, 1]])
    buffer.memories = torch.stack((memory_0, memory_1, memory_2, memory_3), dim=0).unsqueeze(3).swapaxes(1, 2)
    in_0 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    in_1 = torch.tensor([[0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0]])
    in_2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    in_3 = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]])
    buffer.in_episode = torch.stack((in_0, in_1, in_2, in_3), dim=0).unsqueeze(3).swapaxes(1, 2)
    buffer.timestep = torch.tensor([obs_0[0], obs_1[0], obs_2[0], obs_3[0]])
    
    buffer.prepare_batch_dict()
    p_index = 1

    for mini_batch in buffer.mini_batch_generator():
        print("obs")
        print(mini_batch["obs"].shape)
        print(mini_batch["obs"][p_index])
        print("---------------")
        print("memories")
        print(mini_batch["memories"].shape)
        print(mini_batch["memories"].swapaxes(1, 2)[p_index, 0].squeeze())
        print("---------------")
        print("memory mask")
        print(mini_batch["memory_mask"].shape)
        print(mini_batch["memory_mask"][p_index])
        # exit()
        assert torch.equal(torch.sum(mini_batch["memory_mask"], 1), mini_batch["obs"].squeeze())


if __name__ == "__main__":
    main()
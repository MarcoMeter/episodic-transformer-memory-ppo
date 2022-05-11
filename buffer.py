from gym import spaces
import torch
import numpy as np

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, max_episode_length:int, device:torch.device) -> None:
        """
        Args:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
            device {torch.device} -- The device that will be used for training
        """
        # Setup members
        self.device = device
        self.n_workers = config["n_workers"]
        self.worker_steps = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        self.max_episode_length = max_episode_length
        self.num_mem_layers = config["episodic_memory"]["num_layers"]
        self.mem_layer_size = config["episodic_memory"]["layer_size"]

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))
        # Episodic memory buffer tensors
        self.memories = torch.zeros((self.n_workers, self.worker_steps, self.num_mem_layers, self.mem_layer_size), dtype=torch.float32)
        self.in_episode = torch.zeros((self.n_workers, max_episode_length, self.num_mem_layers, self.mem_layer_size), dtype=torch.float32)
        self.out_episode = torch.zeros((self.n_workers, max_episode_length, self.num_mem_layers, self.mem_layer_size), dtype=torch.float32)
        self.timestep = torch.zeros((self.n_workers, ), dtype=torch.uint8)
        self.index_mask = torch.ones((self.n_workers, self.worker_steps), dtype=torch.long)
        self.index = torch.arange(0, self.n_workers * self.worker_steps).reshape(self.n_workers, self.worker_steps).long()

        # Generate episodic memory mask
        self.memory_mask = torch.tril(torch.ones((max_episode_length, max_episode_length)))
        # Shift mask by one to account for the fact that for the first timestep the memory is empty
        self.memory_mask = torch.cat((torch.zeros((1, max_episode_length)), self.memory_mask))[:-1]

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "actions": self.actions,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "obs": self.obs
        }
        # Episodic memory buffer
        episodic_memory = {
            "indices": self.index,
            "index_mask": self.index_mask,
            "memories": self.memories
        }

        # Retrieve the indices of dones as these are the last step of a whole episode
        episode_done_indices = []
        for w in range(self.n_workers):
            episode_done_indices.append(list(self.dones[w].nonzero()[0]))
            # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
            if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.worker_steps - 1:
                episode_done_indices[w].append(self.worker_steps - 1)

        # Process episodic memory to construct full memory episodes and store them in samples
        for key, value in episodic_memory.items():
            episodes = []
            for w in range(self.n_workers):
                start_index = 0
                for count, done_index in enumerate(episode_done_indices[w]):
                    if count == 0 and key == "memories" and self.timestep[w] > 0:
                        # Concat buffer in episode and memories until done index
                        episode = torch.cat((self.in_episode[w, 0:self.timestep[w]], self.memories[w, 0:done_index + 1]))
                    elif count == 0 and (key == "indices" or key == "index_mask") and self.timestep[w] > 0:
                        # Padd the indices and index mask from left to adapt to the episode start from the previous update
                        padding = torch.zeros((self.timestep[w]), dtype=torch.long)
                        episode = torch.cat((padding, value[w, start_index:done_index + 1]))                   
                    else:
                        episode = value[w, start_index:done_index + 1]
                        
                    # Pad the episode with zeros if it is shorter than the maximum episode length
                    episode = self.pad_sequence(episode, self.max_episode_length) # TODO: Maybe pad the episode with its actual longest episode length if it is shorter than the hyperparameter: maximum episode length
                                                                                  # Note: Then we have to regenerate the memory_mask in this method instead in the init function
                    # Append the episode to the list of episodes
                    episodes.append(episode)
                    
                    # Set the start index to the beginning of the next episode
                    start_index = done_index + 1
            samples[key] = torch.stack(episodes)

        # Flatten all samples and convert them to a tensor except memories and its memory mask
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "memories":
                self.samples_flat[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            else:
                self.samples_flat[key] = value
        # Add the memory mask to samples
        self.samples_flat["memory_mask"] = self.memory_mask

    def pad_sequence(self, sequence:torch.tensor, target_length:int) -> np.ndarray:
        """Pads a sequence to the target length using zeros.

        Args:
            sequence {torch.tensor} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
            
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)

    def mini_batch_generator(self):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.

        Arguments:
            num_mini_batches {int} -- Number of the to be sampled mini batches

        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices
        batch_size_with_padding = self.samples_flat["index_mask"].shape[0]
        # Create the indices for the mini batches
        indices = torch.arange(0, batch_size_with_padding).long()
        # Mask the indices that are not part of an episode
        indices = indices[self.samples_flat["index_mask"] == 1]
        # Shuffle the indices
        indices = indices[torch.randperm(self.batch_size)]
        mini_batch_size = self.batch_size // self.n_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "memories":
                    mini_batch_indices_ = torch.floor(mini_batch_indices / self.max_episode_length)
                    mini_batch["memories"] = value[mini_batch_indices_.long()].to(self.device)
                elif key == "memory_mask":
                    mini_batch_indices_ = torch.remainder(mini_batch_indices, self.max_episode_length)
                    mini_batch["memory_mask"] = value[mini_batch_indices_.long()].to(self.device)
                else:
                    mini_batch_indices_ = self.samples_flat["indices"][mini_batch_indices]
                    mini_batch[key] = value[mini_batch_indices_].to(self.device)
                    
            yield mini_batch

    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]
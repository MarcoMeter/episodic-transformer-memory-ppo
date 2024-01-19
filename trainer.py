import numpy as np
import os
import pickle
import time
import torch

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import Agent
from utils import create_env, process_episode_info
from worker import Worker

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set members
        self.config = config
        self.device = device
        self.run_id = run_id
        self.num_workers = config["n_workers"]
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks = config["transformer"]["num_blocks"]
        self.embed_dim = config["transformer"]["embed_dim"]

        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment to retrieve action space, observation space and max episode length
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["environment"])
        observation_space = dummy_env.observation_space
        self.action_space_shape = (dummy_env.action_space.n,)
        self.max_episode_length = dummy_env.max_episode_steps
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.worker_steps = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size = self.num_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks = config["transformer"]["num_blocks"]
        self.embed_dim = config["transformer"]["embed_dim"]

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.num_workers, self.worker_steps, len(self.action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.num_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros((self.num_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.num_workers, self.worker_steps, len(self.action_space_shape)))
        self.values = torch.zeros((self.num_workers, self.worker_steps))
        self.advantages = torch.zeros((self.num_workers, self.worker_steps))
        # Episodic memory index buffer
        # Whole episode memories
        # The length of memories is equal to the number of sampled episodes during training data sampling
        # Each element is of shape (max_episode_length, num_blocks, embed_dim)
        self.stored_memories = []
        # Memory mask used during attention
        self.stored_memory_masks = torch.zeros((self.num_workers, self.worker_steps, self.memory_length), dtype=torch.bool)
        # Index to select the correct episode memory from self.memories
        self.stored_memory_index = torch.zeros((self.num_workers, self.worker_steps), dtype=torch.long)
        # Indices to slice the memory window
        self.stored_memory_indices = torch.zeros((self.num_workers, self.worker_steps, self.memory_length), dtype=torch.long)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = Agent(self.config, observation_space, self.action_space_shape, self.max_episode_length).to(device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["learning_rate"])

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.num_workers)]
        self.worker_ids = range(self.num_workers)
        self.worker_current_episode_step = torch.zeros((self.num_workers, ), dtype=torch.long)
        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        self.next_obs = np.zeros((self.num_workers,) + observation_space.shape, dtype=np.float32)
        for w, worker in enumerate(self.workers):
            self.next_obs[w] = worker.child.recv()

        # Setup placeholders for each worker's current episodic memory
        self.next_memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
        # Generate episodic memory mask used in attention
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """         
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """

        print("Step 6: Starting training using " + str(self.device))
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            # Sample training data
            sampled_episode_info = []
        
            # Init episodic memory buffer using each workers' current episodic memory
            self.stored_memories = [self.next_memory[w] for w in range(self.num_workers)]
            for w in range(self.num_workers):
                self.stored_memory_index[w] = w

            # Sample actions from the model and collect experiences for optimization
            for t in range(self.config["worker_steps"]):
                # Gradients can be omitted for sampling training data
                with torch.no_grad():
                    # Store the initial observations inside the buffer
                    self.obs[:, t] = torch.tensor(self.next_obs)
                    # Store mask and memory indices inside the buffer
                    self.stored_memory_masks[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
                    self.stored_memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
                    # Retrieve the memory window from the entire episodic memory
                    memory_window = batched_index_select(self.next_memory, 1, self.stored_memory_indices[:,t])
                    # Forward the model to retrieve the policy, the states' value and the new memory item
                    policy, value, memory = self.model(torch.tensor(self.next_obs), memory_window, self.stored_memory_masks[:, t],
                                                    self.stored_memory_indices[:,t])
                    
                    # Add new memory item to the episodic memory
                    self.next_memory[self.worker_ids, self.worker_current_episode_step] = memory

                    # Sample actions from each individual policy branch
                    actions = []
                    log_probs = []
                    for action_branch in policy:
                        action = action_branch.sample()
                        actions.append(action)
                        log_probs.append(action_branch.log_prob(action))
                    # Write actions, log_probs and values to buffer
                    self.actions[:, t] = torch.stack(actions, dim=1)
                    self.log_probs[:, t] = torch.stack(log_probs, dim=1)
                    self.values[:, t] = value

                # Send actions to the environments
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", self.actions[w, t].cpu().numpy()))

                # Retrieve step results from the environments
                for w, worker in enumerate(self.workers):
                    obs, self.rewards[w, t], self.dones[w, t], info = worker.child.recv()
                    if info: # i.e. done
                        # Reset the worker's current timestep
                        self.worker_current_episode_step[w] = 0
                        # Store the information of the completed episode (e.g. total reward, episode length)
                        sampled_episode_info.append(info)
                        # Reset the agent (potential interface for providing reset parameters)
                        worker.child.send(("reset", None))
                        # Get data from reset
                        obs = worker.child.recv()
                        # Break the reference to the worker's memory
                        mem_index = self.stored_memory_index[w, t]
                        self.stored_memories[mem_index] = self.stored_memories[mem_index].clone()
                        # Reset episodic memory
                        self.next_memory[w] = torch.zeros((self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
                        if t < self.config["worker_steps"] - 1:
                            # Store memory inside the buffer
                            self.stored_memories.append(self.next_memory[w])
                            # Store the reference of to the current episodic memory inside the buffer
                            self.stored_memory_index[w, t + 1:] = len(self.stored_memories) - 1
                    else:
                        # Increment worker timestep
                        self.worker_current_episode_step[w] +=1
                    # Store latest observations
                    self.next_obs[w] = obs
                                
            # Compute the last value of the current observation and memory window to compute GAE
            start = torch.clip(self.worker_current_episode_step - self.memory_length, 0)
            end = torch.clip(self.worker_current_episode_step, self.memory_length)
            indices = torch.stack([torch.arange(start[b],end[b]) for b in range(self.num_workers)]).long()
            memory_window = batched_index_select(self.next_memory, 1, indices) # Retrieve the memory window from the entire episode
            _, last_value, _ = self.model(torch.tensor(self.next_obs),
                                            memory_window, self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                            self.stored_memory_indices[:,-1])

            # Compute advantages
            with torch.no_grad():
                last_advantage = 0
                mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
                rewards = torch.tensor(self.rewards)
                for t in reversed(range(self.config["worker_steps"])):
                    last_value = last_value * mask[:, t]
                    last_advantage = last_advantage * mask[:, t]
                    delta = rewards[:, t] + self.config["gamma"] * last_value - self.values[:, t]
                    last_advantage = delta + self.config["gamma"] * self.config["lamda"] * last_advantage
                    self.advantages[:, t] = last_advantage
                    last_value = self.values[:, t]

            # Prepare the sampled data inside the buffer (splits data into sequences)
            b_obs = self.obs.reshape(-1, *self.obs.shape[2:])
            b_logprobs = self.log_probs.reshape(-1, *self.log_probs.shape[2:])
            b_actions = self.actions.reshape(-1, *self.actions.shape[2:])
            b_advantages = self.advantages.reshape(-1)
            b_values = self.values.reshape(-1)
            b_memory_index = self.stored_memory_index.reshape(-1)
            b_memory_indices = self.stored_memory_indices.reshape(-1, *self.stored_memory_indices.shape[2:])
            b_memory_mask = self.stored_memory_masks.reshape(-1, *self.stored_memory_masks.shape[2:])
            self.stored_memories = torch.stack(self.stored_memories, dim=0)

            # Train epochs
            train_info = []
            for epoch in range(self.config["epochs"]):
                b_inds = torch.randperm(self.batch_size)
                for start in range(0, self.batch_size, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    mb_inds = b_inds[start: end]
                    mb_memories = self.stored_memories[b_memory_index[mb_inds]]

                    # Select episodic memory windows
                    memory = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])
                    # Forward model
                    policy, value, _ = self.model(b_obs[mb_inds], memory, b_memory_mask[mb_inds], b_memory_indices[mb_inds])

                    # Retrieve and process log_probs from each policy branch
                    log_probs, entropies = [], []
                    for i, policy_branch in enumerate(policy):
                        log_probs.append(policy_branch.log_prob(b_actions[mb_inds][:, i]))
                        entropies.append(policy_branch.entropy())
                    log_probs = torch.stack(log_probs, dim=1)
                    entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

                    # Compute policy surrogates to establish the policy loss
                    normalized_advantage = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                    normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
                    log_ratio = log_probs - b_logprobs[mb_inds]
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * normalized_advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.config["clip_range"], 1.0 + self.config["clip_range"]) * normalized_advantage
                    policy_loss = torch.min(surr1, surr2)
                    policy_loss = policy_loss.mean()

                    # Value  function loss
                    sampled_return = b_values[mb_inds] + b_advantages[mb_inds]
                    clipped_value = b_values[mb_inds] + (value - b_values[mb_inds]).clamp(min=-self.config["clip_range"], max=self.config["clip_range"])
                    vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
                    vf_loss = vf_loss.mean()

                    # Entropy Bonus
                    entropy_bonus = entropies.mean()

                    # Complete loss
                    loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + self.config["beta"] * entropy_bonus)

                    # Compute gradients
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.config["learning_rate"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
                    self.optimizer.step()

                    # Monitor additional training stats
                    approx_kl = (ratio - 1.0) - log_ratio # http://joschu.net/blog/kl-approx.html
                    clip_fraction = (abs((ratio - 1.0)) > self.config["clip_range"]).float().mean()

                    train_info.append([policy_loss.cpu().data.numpy(),
                            vf_loss.cpu().data.numpy(),
                            loss.cpu().data.numpy(),
                            entropy_bonus.cpu().data.numpy(),
                            approx_kl.mean().cpu().data.numpy(),
                            clip_fraction.cpu().data.numpy()])

            training_stats = np.mean(train_info, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = process_episode_info(episode_infos)

            # Print training statistics
            if "success" in episode_result:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.values), torch.mean(self.advantages))
            else:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.values), torch.mean(self.advantages))
            print(result)

            # Write training statistics to tensorboard
            if episode_result:
                for key in episode_result:
                    if "std" not in key:
                        self.writer.add_scalar("episode/" + key, episode_result[key], update)
            self.writer.add_scalar("losses/loss", training_stats[2], update)
            self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
            self.writer.add_scalar("losses/value_loss", training_stats[1], update)
            self.writer.add_scalar("losses/entropy", training_stats[3], update)
            self.writer.add_scalar("training/value_mean", torch.mean(self.values), update)
            self.writer.add_scalar("training/advantage_mean", torch.mean(self.advantages), update)
            self.writer.add_scalar("other/clip_fraction", training_stats[4], update)
            self.writer.add_scalar("other/kl", training_stats[5], update)

        # Save the trained model at the end of the training
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id + ".nn")

        # Close    
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)
import os
import time
import torch
from docopt import docopt
# import tyro
from yaml_parser import YamlParser
from dataclasses import dataclass

import numpy as np
import pickle

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import Agent
from utils import create_env, process_episode_info
from worker import Worker

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MysteryPath-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.75e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    trxl_num_blocks: int = 3
    """the number of transformer blocks"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 96
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = "absolute"
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

if __name__ == "__main__":
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()

    # Determine the device to be used for training and set the default tensor type
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Set members
    num_workers = config["n_workers"]
    memory_length = config["transformer"]["memory_length"]
    num_blocks = config["transformer"]["num_blocks"]
    embed_dim = config["transformer"]["embed_dim"]

    # Setup Tensorboard Summary Writer
    if not os.path.exists("./summaries"):
        os.makedirs("./summaries")
    timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
    writer = SummaryWriter("./summaries/" + run_id + timestamp)

    # Init dummy environment to retrieve action space, observation space and max episode length
    print("Step 1: Init dummy environment")
    dummy_env = create_env(config["environment"])
    observation_space = dummy_env.observation_space
    action_space_shape = (dummy_env.action_space.n,)
    max_episode_length = dummy_env.max_episode_steps
    dummy_env.close()

    # Init buffer
    print("Step 2: Init buffer")
    worker_steps = config["worker_steps"]
    n_mini_batches = config["n_mini_batch"]
    batch_size = num_workers * worker_steps
    mini_batch_size = batch_size // n_mini_batches
    memory_length = config["transformer"]["memory_length"]
    num_blocks = config["transformer"]["num_blocks"]
    embed_dim = config["transformer"]["embed_dim"]

    # Initialize the buffer's data storage
    rewards = np.zeros((num_workers, worker_steps), dtype=np.float32)
    actions = torch.zeros((num_workers, worker_steps, len(action_space_shape)), dtype=torch.long)
    dones = np.zeros((num_workers, worker_steps), dtype=np.bool)
    obs = torch.zeros((num_workers, worker_steps) + observation_space.shape)
    log_probs = torch.zeros((num_workers, worker_steps, len(action_space_shape)))
    values = torch.zeros((num_workers, worker_steps))
    advantages = torch.zeros((num_workers, worker_steps))
    # Episodic memory index buffer
    # Whole episode memories
    # The length of memories is equal to the number of sampled episodes during training data sampling
    # Each element is of shape (max_episode_length, num_blocks, embed_dim)
    stored_memories = []
    # Memory mask used during attention
    stored_memory_masks = torch.zeros((num_workers, worker_steps, memory_length), dtype=torch.bool)
    # Index to select the correct episode memory from memories
    stored_memory_index = torch.zeros((num_workers, worker_steps), dtype=torch.long)
    # Indices to slice the memory window
    stored_memory_indices = torch.zeros((num_workers, worker_steps, memory_length), dtype=torch.long)

    # Init model
    print("Step 3: Init model and optimizer")
    model = Agent(config, observation_space, action_space_shape, max_episode_length).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Init workers
    print("Step 4: Init environment workers")
    workers = [Worker(config["environment"]) for w in range(num_workers)]
    worker_ids = range(num_workers)
    worker_current_episode_step = torch.zeros((num_workers, ), dtype=torch.long)
    # Reset workers (i.e. environments)
    print("Step 5: Reset workers")
    for worker in workers:
        worker.child.send(("reset", None))
    # Grab initial observations and store them in their respective placeholder location
    next_obs = np.zeros((num_workers,) + observation_space.shape, dtype=np.float32)
    for w, worker in enumerate(workers):
        next_obs[w] = worker.child.recv()

    # Setup placeholders for each worker's current episodic memory
    next_memory = torch.zeros((num_workers, max_episode_length, num_blocks, embed_dim), dtype=torch.float32)
    # Generate episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((memory_length, memory_length)), diagonal=-1)
    """ e.g. memory mask tensor looks like this if memory_length = 6
    0, 0, 0, 0, 0, 0
    1, 0, 0, 0, 0, 0
    1, 1, 0, 0, 0, 0
    1, 1, 1, 0, 0, 0
    1, 1, 1, 1, 0, 0
    1, 1, 1, 1, 1, 0
    """         
    # Setup memory window indices to support a sliding window over the episodic memory
    repetitions = torch.repeat_interleave(torch.arange(0, memory_length).unsqueeze(0), memory_length - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + memory_length) for i in range(max_episode_length - memory_length + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    1, 2, 3, 4
    2, 3, 4, 5
    3, 4, 5, 6
    """

    print("Step 6: Starting training using " + str(device))
    # Store episode results for monitoring statistics
    episode_infos = deque(maxlen=100)

    for update in range(config["updates"]):
        # Sample training data
        sampled_episode_info = []
    
        # Init episodic memory buffer using each workers' current episodic memory
        stored_memories = [next_memory[w] for w in range(num_workers)]
        for w in range(num_workers):
            stored_memory_index[w] = w

        # Sample actions from the model and collect experiences for optimization
        for step in range(config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Store the initial observations inside the buffer
                obs[:, step] = torch.tensor(next_obs)
                # Store mask and memory indices inside the buffer
                stored_memory_masks[:, step] = memory_mask[torch.clip(worker_current_episode_step, 0, memory_length - 1)]
                stored_memory_indices[:, step] = memory_indices[worker_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[:,step])
                # Forward the model to retrieve the policy, the states' value and the new memory item
                policy, value, memory = model(torch.tensor(next_obs), memory_window, stored_memory_masks[:, step],
                                                stored_memory_indices[:,step])
                
                # Add new memory item to the episodic memory
                next_memory[worker_ids, worker_current_episode_step] = memory

                # Sample actions from each individual policy branch
                action = []
                log_prob = []
                for action_branch in policy:
                    a = action_branch.sample()
                    action.append(a)
                    log_prob.append(action_branch.log_prob(a))
                # Write actions, log_probs and values to buffer
                actions[:, step] = torch.stack(action, dim=1)
                log_probs[:, step] = torch.stack(log_prob, dim=1)
                values[:, step] = value

            # Send actions to the environments
            for w, worker in enumerate(workers):
                worker.child.send(("step", actions[w, step].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(workers):
                o, rewards[w, step], dones[w, step], info = worker.child.recv()
                if info: # i.e. done
                    # Reset the worker's current timestep
                    worker_current_episode_step[w] = 0
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    sampled_episode_info.append(info)
                    # Reset the agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    o = worker.child.recv()
                    # Break the reference to the worker's memory
                    mem_index = stored_memory_index[w, step]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    # Reset episodic memory
                    next_memory[w] = torch.zeros((max_episode_length, num_blocks, embed_dim), dtype=torch.float32)
                    if step < config["worker_steps"] - 1:
                        # Store memory inside the buffer
                        stored_memories.append(next_memory[w])
                        # Store the reference of to the current episodic memory inside the buffer
                        stored_memory_index[w, step + 1:] = len(stored_memories) - 1
                else:
                    # Increment worker timestep
                    worker_current_episode_step[w] +=1
                # Store latest observations
                next_obs[w] = o
                            
        # Compute the last value of the current observation and memory window to compute GAE
        start = torch.clip(worker_current_episode_step - memory_length, 0)
        end = torch.clip(worker_current_episode_step, memory_length)
        indices = torch.stack([torch.arange(start[b],end[b]) for b in range(num_workers)]).long()
        memory_window = batched_index_select(next_memory, 1, indices) # Retrieve the memory window from the entire episode
        _, last_value, _ = model(torch.tensor(next_obs),
                                        memory_window, memory_mask[torch.clip(worker_current_episode_step, 0, memory_length - 1)],
                                        stored_memory_indices[:,-1])

        # Compute advantages
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(rewards)
            for t in reversed(range(config["worker_steps"])):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + config["gamma"] * last_value - values[:, t]
                last_advantage = delta + config["gamma"] * config["lamda"] * last_advantage
                advantages[:, t] = last_advantage
                last_value = values[:, t]

        # Prepare the sampled data inside the buffer (splits data into sequences)
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
        b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
        stored_memories = torch.stack(stored_memories, dim=0)

        # Train epochs
        train_info = []
        for epoch in range(config["epochs"]):
            b_inds = torch.randperm(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start: end]
                mb_memories = stored_memories[b_memory_index[mb_inds]]

                # Select episodic memory windows
                memory = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])
                # Forward model
                policy, value, _ = model(b_obs[mb_inds], memory, b_memory_mask[mb_inds], b_memory_indices[mb_inds])

                # Retrieve and process log_probs from each policy branch
                logprobs, entropies = [], []
                for i, policy_branch in enumerate(policy):
                    logprobs.append(policy_branch.log_prob(b_actions[mb_inds][:, i]))
                    entropies.append(policy_branch.entropy())
                logprobs = torch.stack(logprobs, dim=1)
                entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

                # Compute policy surrogates to establish the policy loss
                normalized_advantage = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(action_space_shape)) # Repeat is necessary for multi-discrete action spaces
                log_ratio = logprobs - b_logprobs[mb_inds]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * normalized_advantage
                surr2 = torch.clamp(ratio, 1.0 - config["clip_range"], 1.0 + config["clip_range"]) * normalized_advantage
                policy_loss = torch.min(surr1, surr2)
                policy_loss = policy_loss.mean()

                # Value  function loss
                sampled_return = b_values[mb_inds] + b_advantages[mb_inds]
                clipped_value = b_values[mb_inds] + (value - b_values[mb_inds]).clamp(min=-config["clip_range"], max=config["clip_range"])
                vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
                vf_loss = vf_loss.mean()

                # Entropy Bonus
                entropy_bonus = entropies.mean()

                # Complete loss
                loss = -(policy_loss - config["value_loss_coefficient"] * vf_loss + config["beta"] * entropy_bonus)

                # Compute gradients
                for pg in optimizer.param_groups:
                    pg["lr"] = config["learning_rate"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_grad_norm"])
                optimizer.step()

                # Monitor additional training stats
                approx_kl = (ratio - 1.0) - log_ratio # http://joschu.net/blog/kl-approx.html
                clip_fraction = (abs((ratio - 1.0)) > config["clip_range"]).float().mean()

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
                training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(values), torch.mean(advantages))
        else:
            result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(values), torch.mean(advantages))
        print(result)

        # Write training statistics to tensorboard
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    writer.add_scalar("episode/" + key, episode_result[key], update)
        writer.add_scalar("losses/loss", training_stats[2], update)
        writer.add_scalar("losses/policy_loss", training_stats[0], update)
        writer.add_scalar("losses/value_loss", training_stats[1], update)
        writer.add_scalar("losses/entropy", training_stats[3], update)
        writer.add_scalar("training/value_mean", torch.mean(values), update)
        writer.add_scalar("training/advantage_mean", torch.mean(advantages), update)
        writer.add_scalar("other/clip_fraction", training_stats[4], update)
        writer.add_scalar("other/kl", training_stats[5], update)

    # Save the trained model at the end of the training
    if not os.path.exists("./models"):
        os.makedirs("./models")
    model.cpu()
    pickle.dump((model.state_dict(), config), open("./models/" + run_id + ".nn", "wb"))
    print("Model saved to " + "./models/" + run_id + ".nn")

    # Close    
    try:
        dummy_env.close()
    except:
        pass

    try:
        writer.close()
    except:
        pass

    try:
        for worker in workers:
            worker.child.send(("close", None))
    except:
        pass

    time.sleep(1.0)
    exit(0)

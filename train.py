import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F
import tyro

from einops import rearrange
from collections import deque
from torch.utils.tensorboard import SummaryWriter

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
    env_type: str = "PocMemoryEnv"
    """test"""
    env_id: str = "MysteryPath-Grid-v0" # CartPoleMasked CartPoleMasked MiniGrid-MemoryS9-v0 MysteryPath-Grid-v0 MortarMayhem-Grid-v0
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_blocks: int = 3
    """the number of transformer blocks"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 32
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = ""
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb
    
class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        # Get number of training examples and sequence lengths
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax wo retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )

        # Forward projection
        out = self.fc_out(out)

        return out, attention
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value
        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)
        # Skip connection
        x = attention + query
        # Pre-layer normalization
        x_ = self.layer_norm_attn(x)
        # Forward projection
        forward = self.fc_projection(x_)
        # Skip connection
        out = forward + x
        return out, attention_weights

class Transformer(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, max_episode_steps, positional_encoding):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_blocks)])

    def forward(self, x, memories, mask, memory_indices):
        # Add positional encoding to every transformer block input
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        # Forward transformer blocks and return new memories (i.e. hidden states)
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(x.detach())
            x, attention_weights = block(memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask) # args: value, key, query, mask
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)
    
class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, max_episode_steps):
        super().__init__()
        self.observation_space_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.observation_space_shape) > 1:
            self.cnn = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 384)),
                nn.ReLU(),
            )
            in_features_next_layer = 384
        else:
            in_features_next_layer = observation_space.shape[0]

        self.lin_hidden = layer_init(nn.Linear(in_features_next_layer, args.trxl_dim))

        self.transformer = Transformer(args.trxl_num_blocks, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding)

        self.actor_branches = nn.ModuleList([
            layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
            for num_actions in action_space_shape
        ])
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)
    
    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.observation_space_shape) > 1:
            x = self.cnn(x)
        x = F.relu(self.lin_hidden(x))
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        if len(self.observation_space_shape) > 1:
            x = self.cnn(x)
        x = F.relu(self.lin_hidden(x))
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
        log_probs = []
        for i, dist in enumerate(probs):
            log_probs.append(dist.log_prob(action[:, i]))
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1)
        return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), memory

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Determine the device to be used for training and set the default tensor type
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Setup Tensorboard Summary Writer
    if not os.path.exists("./summaries"):
        os.makedirs("./summaries")
    timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
    writer = SummaryWriter("./summaries/" + "x_" + timestamp)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    random.SystemRandom().seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Init dummy environment to retrieve action space, observation space and max episode length
    print("Step 1: Init dummy environment")
    dummy_env = create_env(args)
    observation_space = dummy_env.observation_space
    action_space_shape = (dummy_env.action_space.n,)
    max_episode_steps = dummy_env.max_episode_steps
    dummy_env.close()

    # Init buffer fields
    print("Step 2: Init buffer")
    rewards = torch.zeros((args.num_steps, args.num_envs))
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
    dones = torch.zeros((args.num_steps, args.num_envs))
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
    log_probs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)))
    values = torch.zeros((args.num_steps, args.num_envs))
    # Episodic memory index buffer
    # Whole episode memories
    # The length of memories is equal to the number of sampled episodes during training data sampling
    # Each element is of shape (max_episode_length, num_blocks, embed_dim)
    stored_memories = []
    # Memory mask used during attention
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    # Index to select the correct episode memory from memories
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    # Indices to slice the memory window
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    # Init model
    print("Step 3: Init model and optimizer")
    agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
    agent.train()
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate)

    # Init workers
    print("Step 4: Init environment workers")
    workers = [Worker(args) for w in range(args.num_envs)]
    worker_ids = range(args.num_envs)
    worker_current_episode_step = torch.zeros((args.num_envs, ), dtype=torch.long)
    # Reset workers (i.e. environments)
    print("Step 5: Reset workers")
    for worker in workers:
        worker.child.send(("reset", None))
    # Grab initial observations and store them in their respective placeholder location
    next_obs = np.zeros((args.num_envs,) + observation_space.shape, dtype=np.float32)
    next_done = torch.zeros(args.num_envs)
    for w, worker in enumerate(workers):
        next_obs[w] = worker.child.recv()

    # Setup placeholders for each worker's current episodic memory
    next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_blocks, args.trxl_dim), dtype=torch.float32)
    # Generate episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
    """ e.g. memory mask tensor looks like this if memory_length = 6
    0, 0, 0, 0, 0, 0
    1, 0, 0, 0, 0, 0
    1, 1, 0, 0, 0, 0
    1, 1, 1, 0, 0, 0
    1, 1, 1, 1, 0, 0
    1, 1, 1, 1, 1, 0
    """         
    # Setup memory window indices to support a sliding window over the episodic memory
    repetitions = torch.repeat_interleave(torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]).long()
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

    for iteration in range(1, args.num_iterations + 1):
        # Sample training data
        sampled_episode_info = []
    
        # Init episodic memory buffer using each workers' current episodic memory
        stored_memories = [next_memory[w] for w in range(args.num_envs)]
        for w in range(args.num_envs):
            stored_memory_index[:, w] = w

        # Sample actions from the model and collect experiences for optimization
        for step in range(args.num_steps):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Store the initial observations inside the buffer
                obs[step] = torch.tensor(next_obs)
                dones[step] = next_done
                # Store mask and memory indices inside the buffer
                stored_memory_masks[step] = memory_mask[torch.clip(worker_current_episode_step, 0, args.trxl_memory_length - 1)]
                stored_memory_indices[step] = memory_indices[worker_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                # Forward the model to retrieve the policy, the states' value and the new memory item
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    torch.tensor(next_obs), memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                
                # Add new memory item to the episodic memory
                next_memory[worker_ids, worker_current_episode_step] = new_memory
                actions[step] = action
                log_probs[step] = logprob
                values[step] = value

            # Send actions to the environments
            for w, worker in enumerate(workers):
                worker.child.send(("step", actions[step, w].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(workers):
                o, rewards[step, w], next_done[w], info = worker.child.recv()
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
                    mem_index = stored_memory_index[step, w]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    # Reset episodic memory
                    next_memory[w] = torch.zeros((max_episode_steps, args.trxl_num_blocks, args.trxl_dim), dtype=torch.float32)
                    if step < args.num_steps - 1:
                        # Store memory inside the buffer
                        stored_memories.append(next_memory[w])
                        # Store the reference of to the current episodic memory inside the buffer
                        stored_memory_index[step + 1:, w] = len(stored_memories) - 1
                else:
                    # Increment worker timestep
                    worker_current_episode_step[w] +=1
                # Store latest observations
                next_obs[w] = o
                            
        # Bootstrap value if not done
        with torch.no_grad():
            start = torch.clip(worker_current_episode_step - args.trxl_memory_length, 0)
            end = torch.clip(worker_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start[b],end[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices) # Retrieve the memory window from the entire episode
            next_value = agent.get_value(torch.tensor(next_obs),
                                            memory_window, memory_mask[torch.clip(worker_current_episode_step, 0, args.trxl_memory_length - 1)],
                                            stored_memory_indices[-1])
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
        b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
        stored_memories = torch.stack(stored_memories, dim=0)

        # Optimizing the policy and value network
        train_info = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start: end]
                mb_memories = stored_memories[b_memory_index[mb_inds]]
                mb_memory_windows = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], mb_memory_windows, b_memory_mask[mb_inds], b_memory_indices[mb_inds], b_actions[mb_inds]
                )

                # Policy loss
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages.unsqueeze(1).repeat(1, len(action_space_shape)) # Repeat is necessary for multi-discrete action spaces
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)
                pgloss1 = -mb_advantages * ratio
                pgloss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pgloss1, pgloss2).mean()

                # Value loss
                mb_returns = b_values[mb_inds] + b_advantages[mb_inds]
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                if args.clip_vloss:
                    v_loss_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(min=-args.clip_coef, max=args.clip_coef)
                    v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - mb_returns) ** 2).mean()
                else:
                    v_loss = v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0)) - logratio # http://joschu.net/blog/kl-approx.html
                    clip_fraction = (abs((ratio - 1.0)) > args.clip_coef).float().mean()

                train_info.append([pg_loss.cpu().data.numpy(),
                        v_loss.cpu().data.numpy(),
                        loss.cpu().data.numpy(),
                        entropy_loss.cpu().data.numpy(),
                        approx_kl.mean().cpu().data.numpy(),
                        clip_fraction.cpu().data.numpy()])

        training_stats = np.mean(train_info, axis=0)

        # Store recent episode infos
        episode_infos.extend(sampled_episode_info)
        episode_result = process_episode_info(episode_infos)

        # Print training statistics
        if "success" in episode_result:
            result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                iteration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success"],
                training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(values), torch.mean(advantages))
        else:
            result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                iteration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(values), torch.mean(advantages))
        print(result)

        # Write training statistics to tensorboard
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    writer.add_scalar("episode/" + key, episode_result[key], iteration)
        writer.add_scalar("losses/loss", training_stats[2], iteration)
        writer.add_scalar("losses/policy_loss", training_stats[0], iteration)
        writer.add_scalar("losses/value_loss", training_stats[1], iteration)
        writer.add_scalar("losses/entropy", training_stats[3], iteration)
        writer.add_scalar("training/value_mean", torch.mean(values), iteration)
        writer.add_scalar("training/advantage_mean", torch.mean(advantages), iteration)
        writer.add_scalar("other/clip_fraction", training_stats[5], iteration)
        writer.add_scalar("other/kl", training_stats[4], iteration)
 
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

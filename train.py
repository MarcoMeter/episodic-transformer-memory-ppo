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
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from pom_env import PoMEnv
from torch.utils.tensorboard import SummaryWriter

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
    env_id: str = "ProofofMemory-v0" # MiniGrid-MemoryS9-v0 MysteryPath-Grid-v0 MortarMayhem-Grid-v0 ProofofMemory-v0
    """the id of the environment"""
    total_timesteps: int = 25000
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
    vf_coef: float = 0.1
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_blocks: int = 4
    """the number of transformer blocks"""
    trxl_num_heads: int = 1
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 64
    """the dimension of the transformer"""
    trxl_memory_length: int = 16
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = ""
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """
    reconstruction_coef: float = 0.0
    """the coefficient of the observation reconstruction loss, if set to 0.0 the reconstruction loss is not used"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        if "MiniGrid" in env_id:
            env = gym.make(env_id, agent_view_size=3, tile_size = 28)
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size = 28))

        if len(env.observation_space.shape) > 1:
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 1)
        env = gym.wrappers.TimeLimit(env, 96) # TODO: retrieve from Memory Gym envs
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk

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
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection"""
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
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)
        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax to retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3) # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )

        return self.fc_out(out), attention
    
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
        key = value                     # K = V -> self-attention
        attention, attention_weights = self.attention(value, key, query_, mask) # MHA
        x = attention + query               # Skip connection
        x_ = self.layer_norm_attn(x)        # Pre-layer normalization
        forward = self.fc_projection(x_)    # Forward projection
        out = forward + x                   # Skip connection
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
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.obs_shape) > 1:
            self.cnn = nn.Sequential(
                layer_init(nn.Conv2d(1, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
            in_features_next_layer = args.trxl_dim
        else:
            in_features_next_layer = observation_space.shape[0]

        self.lin_hidden = layer_init(nn.Linear(in_features_next_layer, args.trxl_dim))

        self.transformer = Transformer(args.trxl_num_blocks, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding)

        self.actor_branches = nn.ModuleList([
            layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
            for num_actions in action_space_shape
        ])
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 1, 8, stride=4)),
                nn.Sigmoid(),
            )
    
    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.obs_shape) > 1:
            x = self.cnn(x / 255.0)
        x = F.relu(self.lin_hidden(x))
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        if len(self.obs_shape) > 1:
            x = self.cnn(x / 255.0)
        x = F.relu(self.lin_hidden(x))
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        self.x = x
        probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
        log_probs = []
        for i, dist in enumerate(probs):
            log_probs.append(dist.log_prob(action[:, i]))
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1)
        return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), memory
    
    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x

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

    print("Step 1: Init environments")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    observation_space = envs.single_observation_space
    action_space_shape = (envs.single_action_space.n,) if isinstance(envs.single_action_space, gym.spaces.Discrete) else tuple(envs.single_action_space.nvec)
    max_episode_steps = 96

    print("Step 2: Init buffer")
    rewards = torch.zeros((args.num_steps, args.num_envs))
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
    dones = torch.zeros((args.num_steps, args.num_envs))
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
    log_probs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)))
    values = torch.zeros((args.num_steps, args.num_envs))
    # The length of stored-memories is equal to the number of sampled episodes during training data sampling 
    # (num_episodes, max_episode_length, num_blocks, embed_dim)
    stored_memories = []
    # Memory mask used during attention
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    # Index to select the correct episode memory from stored_memories
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    # Indices to slice the episode memories into windows
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    print("Step 3: Init model and optimizer")
    agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
    agent.train()
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCELoss() # Binary cross entropy loss for observation reconstruction

    print("Step 4: Reset environments")
    # Grab initial observations and store them in their respective placeholder location
    global_step = 0
    env_ids = range(args.num_envs)
    env_current_episode_step = torch.zeros((args.num_envs, ), dtype=torch.long)
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs)

    # Setup placeholders for each environments's current episodic memory
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

    print("Step 5: Starting training using " + str(device))
    episode_infos = deque(maxlen=100)   # Store episode results for monitoring statistics

    for iteration in range(1, args.num_iterations + 1):
        sampled_episode_infos = []
    
        # Init episodic memory buffer using each workers' current episodic memory
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        for step in range(args.num_steps):
            global_step += args.num_envs
            with torch.no_grad():
                obs[step] = next_obs
                dones[step] = next_done
                stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
                stored_memory_indices[step] = memory_indices[env_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                next_memory[env_ids, env_current_episode_step] = new_memory
                actions[step] = action
                log_probs[step] = logprob
                values[step] = value

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Reset and process episodic memory if done
            for id, done in enumerate(next_done):
                if done:
                    # Reset the environment's current timestep
                    env_current_episode_step[id] = 0
                    # Break the reference to the environment's episodic memory
                    mem_index = stored_memory_index[step, id]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    # Reset episodic memory
                    next_memory[id] = torch.zeros((max_episode_steps, args.trxl_num_blocks, args.trxl_dim), dtype=torch.float32)
                    if step < args.num_steps - 1:
                        # Store memory inside the buffer
                        stored_memories.append(next_memory[id])
                        # Store the reference of to the current episodic memory inside the buffer
                        stored_memory_index[step + 1:, id] = len(stored_memories) - 1
                else:
                    # Increment environment timestep if not done
                    env_current_episode_step[id] +=1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        sampled_episode_infos.append(info["episode"])
                            
        # Bootstrap value if not done
        with torch.no_grad():
            start = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start[b],end[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices) # Retrieve the memory window from the entire episode
            next_value = agent.get_value(next_obs,
                                            memory_window, memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
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

                # Entropy loss
                entropy_loss = entropy.mean()

                # Combined losses
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Add reconstruction loss if used
                if args.reconstruction_coef > 0.0:
                    r_loss = bce_loss(agent.reconstruct_observation(), b_obs[mb_inds] / 255.0)
                    loss += args.reconstruction_coef * r_loss

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

        # Log and monitor training statistics
        episode_infos.extend(sampled_episode_infos)
        episode_result = {}
        if len(episode_infos) > 0:
            for key in episode_infos[0].keys():
                episode_result[key + "_mean"] = np.mean([info[key] for info in episode_infos])
                episode_result[key + "_std"] = np.std([info[key] for info in episode_infos])

        result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                iteration, episode_result["r_mean"], episode_result["r_std"], episode_result["l_mean"], episode_result["l_std"], 
                training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(values), torch.mean(advantages))
        print(result)

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
 
    writer.close()
    envs.close()

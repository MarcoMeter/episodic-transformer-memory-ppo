import numpy as np
import torch

from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F
from einops import rearrange

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_steps):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.memory_layer_size = config["transformer"]["embed_dim"]
        self.observation_space_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
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
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]

        self.lin_hidden = layer_init(nn.Linear(in_features_next_layer, self.memory_layer_size))

        self.transformer = Transformer(config["transformer"], self.memory_layer_size, self.max_episode_steps)

        self.actor_branches = nn.ModuleList([
            layer_init(nn.Linear(self.hidden_size, out_features=num_actions), np.sqrt(0.01))
            for num_actions in action_space_shape
        ])
        self.critic = layer_init(nn.Linear(self.hidden_size, 1), 1)

    def forward(self, obs, memory, memory_mask, memory_indices):
        h = obs
        if len(self.observation_space_shape) > 1:
            h = self.cnn(h)

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))
        
        # Forward transformer blocks
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)

        value = self.critic(h).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h)) for branch in self.actor_branches]
        
        return pi, value, memory    

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
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.layer_norm_attn = nn.LayerNorm(embed_dim)
        self.fc_projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value
        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)
        # Skip connection
        h = attention + query
        # Pre-layer normalization
        h_ = self.layer_norm_attn(h)
        # Forward projection
        forward = self.fc_projection(h_)
        # Skip connection
        out = forward + h
        return out, attention_weights

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

class Transformer(nn.Module):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, config, input_dim, max_episode_steps) -> None:
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.max_episode_steps = max_episode_steps
        self.activation = nn.ReLU()

        # Input embedding layer
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Determine positional encoding
        if config["positional_encoding"] == "absolute":
            self.pos_embedding = PositionalEncoding(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices):
        """
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
        """
        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(h))

        # Add positional encoding to every transformer block input
        if self.config["positional_encoding"] == "relative":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
            # memories[:,:,0] = memories[:,:,0] + pos_embedding # add positional encoding only to first layer?
        elif self.config["positional_encoding"] == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)
            # memories[:,:,0] = memories[:,:,0] + self.pos_embedding[memory_indices] # add positional encoding only to first layer?

        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            h, attention_weights = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), mask) # args: value, key, query, mask
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)

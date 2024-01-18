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
        super().__init__()
        conf = config["transformer"]
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

        self.lin_hidden = layer_init(nn.Linear(in_features_next_layer, conf["embed_dim"]))

        self.transformer = Transformer(conf["num_blocks"], conf["embed_dim"], conf["num_heads"], self.max_episode_steps, conf["positional_encoding"])

        self.actor_branches = nn.ModuleList([
            layer_init(nn.Linear(conf["embed_dim"], out_features=num_actions), np.sqrt(0.01))
            for num_actions in action_space_shape
        ])
        self.critic = layer_init(nn.Linear(conf["embed_dim"], 1), 1)

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

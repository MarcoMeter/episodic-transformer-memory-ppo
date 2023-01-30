import numpy as np
import torch
from torch import nn
from einops import rearrange
from neroRL.nn.module import Module

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
        """
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
        """
        The forward pass of the multi head attention layer.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
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
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

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
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        return out, attention
        
class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, attention_norm, projection_norm):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.
        
        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
            attention_norm {str} -- Whether to apply LayerNorm "pre" or "post" attention
            projection_norm {str} -- Whether to apply LayerNorm "pre" or "post" the feed forward projection
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # LayerNorms
        self.attention_norm = attention_norm
        self.projection_norm = projection_norm
        if "qkv" in attention_norm:
            # In the case of just "pre" LayerNorm, only the query is considered
            # Use "pre_qkv" to also apply LayerNorm to keys and values
            self.norm_kv = nn.LayerNorm(embed_dim)
        if "pre" in attention_norm or attention_norm == "post":
            self.norm1 = nn.LayerNorm(embed_dim)
        if projection_norm == "pre" or projection_norm == "post":
            self.norm2 = nn.LayerNorm(embed_dim)

        # Feed forward projection
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Transformer Block forward pass.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Apply pre-layer norm across the attention input
        if "pre" in self.attention_norm:
            query_ = self.norm1(query)
            # Apply layer norm to value and key as well?
            if "qkv" in self.attention_norm:
                value = self.norm_kv(value)
                key = value
        else:
            query_ = query

        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)

        # Add skip connection and run through normalization
        h = attention + query
        # Apply post-layer norm across the attention output (i.e. attention input)
        if self.attention_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.projection_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h

        # Forward projection
        forward = self.fc(h_)

        # Add skip connection and run through normalization
        out = forward + h
        # Apply post-layer norm across the projection output
        if self.projection_norm == "post":
            out = self.norm2(out)
        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
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
        """Sets up the input embedding, positional encoding and the transformer blocks.
        
        Arguments:
            config {dict} -- Transformer config
            input_dim {int} -- Dimension of the input
            activation {torch.nn.modules.activation} -- Activation function of the input embedding
        """
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
        if config["positional_encoding"] == "relative":
            self.pos_embedding = SinusoidalPosition(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config["attention_norm"], config["projection_norm"]) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices):
        """
        Transformer encoder forward pass.
        
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
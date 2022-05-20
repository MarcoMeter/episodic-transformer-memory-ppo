import torch
from torch import QInt32Storage, nn
from einops import rearrange, repeat
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, visualize=False):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.visualize = visualize
        self.head_size = embed_size // num_heads

        assert (
            self.head_size * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)
        # Visualize attention coefficients of worker 0
        # if visualize is True
        if self.visualize:
            self.visualize_coef(attention[0].squeeze(0).squeeze(0), mask[0])

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out
    
    def visualize_coef(self, attention_coef, mask):
        if attention_coef.requires_grad:
            return
        attention_coef = attention_coef.flatten()
        plt.bar(range(len(attention_coef)), attention_coef)
        plt.ylim(top = 1)
        plt.title("Attention Coefficients of worker 0 at step " + str(mask.sum().item()))
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion = 1, visualize_coef=False):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads, visualize_coef)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection and run through normalization
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class SinusoidalPosition(nn.Module):
    def __init__(
        self,
        dim,
        min_timescale = 2.,
        max_timescale = 1e4
    ):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, x):
        seq_len = x.shape[1]
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb
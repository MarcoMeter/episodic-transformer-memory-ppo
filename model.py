import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from transformer import TransformerBlock, SinusoidalPosition

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_length, visualize_coef = False):
        """Model setup

        Args:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.memory_layer_size = config["episodic_memory"]["layer_size"]
        self.num_mem_layers = config["episodic_memory"]["num_layers"]
        self.num_heads = config["episodic_memory"]["num_heads"]
        self.num_mem_layers = config["episodic_memory"]["num_layers"]
        self.observation_space_shape = observation_space.shape
        self.max_episode_length = max_episode_length

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]
        
        # Hidden layer
        self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Transformer Blocks
        self.pos_emb = SinusoidalPosition(dim = self.memory_layer_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.memory_layer_size, self.num_heads, visualize_coef = visualize_coef) 
            for _ in range(self.num_mem_layers)])
        # TODO init weights

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.memory_layer_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy
        self.policy = nn.Linear(self.hidden_size, action_space_shape[0])
        nn.init.orthogonal_(self.policy.weight, np.sqrt(0.01))

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, memories:torch.tensor, memory_mask:torch.tensor):
        """Forward pass of the model

        Args:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Transformer positional encoding
        pos_embedding = self.pos_emb(memories)
        pos_embedding = torch.repeat_interleave(pos_embedding.unsqueeze(1), self.num_mem_layers, dim = 1)
        memories = memories + pos_embedding
        
        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            h = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), memory_mask).squeeze() # args: value, key, query, mask
            if len(h.shape) == 1:
                h = h.unsqueeze(0)

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = Categorical(logits=self.policy(h_policy))

        memories = torch.stack(out_memories, dim=1)
        return pi, value, memories

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Args:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by name
        """
        grads = {}
        if len(self.observation_space_shape) > 1:
            grads["encoder"] = self._calc_grad_norm([self.conv1, self.conv2, self.conv3])  
            
        grads["linear_layer"] = self._calc_grad_norm([self.lin_hidden])
        
        for i, block in enumerate(self.transformer_blocks):
            grads["transformer_block_" + str(i)] = self._calc_grad_norm([block])
             
        grads["policy"] = self._calc_grad_norm([self.lin_policy, self.policy])
        grads["value"] = self._calc_grad_norm([self.lin_value, self.value])
          
        return grads
    
    def _calc_grad_norm(self, modules:list):
        """Computes the norm of the gradients of the given modules.

        Args:
            modules {list}: List of modules to compute the norm of the gradients of.

        Returns:
            {float} -- Norm of the gradients of the given modules. 
        """
        grads = []
        for module in modules:
            for name, parameter in module.named_parameters():
                grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None
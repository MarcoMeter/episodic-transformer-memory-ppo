import torch
from torch import nn
import numpy as np
from environments.cartpole_env import CartPole
from environments.minigrid_env import Minigrid
from environments.poc_memory_env import PocMemoryEnv
from environments.mortar_env import MortarABEnv, MortarBEnv
from environments.memory_gym_env import MemoryGymWrapper

def create_env(config:dict):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        env_name {str}: Name of the to be instantiated environment

    Returns:
        {env}: Returns the selected environment instance.
    """
    if config["env"] == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True, max_episode_steps=32)
    if config["env"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["env"] == "CartPoleMasked":
        return CartPole(mask_velocity=True)
    if config["env"] == "Minigrid":
        return Minigrid(config["name"])
    if config["env"] == "MortarAB":
        return MortarABEnv()
    if config["env"] == "MortarB":
        return MortarBEnv()
    if config["env"] in ["SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"]:
        return MemoryGymWrapper(config["env"] + "-v0")

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)
    
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def process_episode_info(episode_info:list) -> dict:
    """Extracts the mean and std of completed episode statistics like length and total reward.

    Arguments:
        episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

    Returns:
        {dict} -- Processed episode results (computes the mean and std for most available keys)
    """
    result = {}
    if len(episode_info) > 0:
        for key in episode_info[0].keys():
            if key == "success":
                # This concerns the PocMemoryEnv only
                episode_result = [info[key] for info in episode_info]
                result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
            result[key + "_mean"] = np.mean([info[key] for info in episode_info])
            result[key + "_std"] = np.std([info[key] for info in episode_info])
    return result

class Module(nn.Module):
    """nn.Module is extended by functions to compute the norm and the mean of this module's parameters."""
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm.

        Returns:
            {float}: Returns the norm of the gradients of this model's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean.

        Returns:
            {float}: Returns the mean of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None
from environments.cartpole_env import CartPole
from environments.minigrid_env import Minigrid
from environments.poc_memory_env import PocMemoryEnv
from environments.mortar_env import MortarABEnv, MortarBEnv

def create_env(config:dict):
    """Initializes an environment based on the provided environment name.
    
    Args:
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

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Args:
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
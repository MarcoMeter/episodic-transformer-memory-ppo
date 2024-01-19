import numpy as np

from environments.cartpole_env import CartPole
from environments.memory_gym_env import MemoryGymWrapper
from environments.minigrid_env import Minigrid
from environments.poc_memory_env import PocMemoryEnv

def create_env(args, render:bool=False):
    if args.env_type == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True, max_episode_steps=32)
    if args.env_type == "CartPole":
        return CartPole(mask_velocity=False)
    if args.env_type == "CartPoleMasked":
        return CartPole(mask_velocity=True)
    if args.env_type == "Minigrid":
        return Minigrid(args.env_id)
    if args.env_type in ["SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"]:
        return MemoryGymWrapper(env_name = args.env_id, reset_params={}, realtime_mode=render)

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

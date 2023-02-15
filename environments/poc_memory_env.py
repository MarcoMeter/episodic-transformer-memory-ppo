from gym.spaces import space
import numpy as np
from gym import spaces
import time
import os
from reprint import output

class PocMemoryEnv():
    """
    Proof of Concept Memory Environment

    This environment is intended to assess whether the underlying recurrent policy is working or not.
    The environment is based on a one dimensional grid where the agent can move left or right.
    At both ends, a goal is spawned that is either punishing or rewarding.
    During the very first two steps, the agent gets to know which goal leads to a positive or negative reward.
    Afterwards, this information is hidden in the agent's observation.
    The last value of the agent's observation is its current position inside the environment.
    Optionally and to increase the difficulty of the task, the agent's position can be frozen until the goal information is hidden.
    To further challenge the agent, the step_size can be decreased.
    """
    def __init__(self, step_size:float=0.2, glob:bool=False, freeze:bool=False, max_episode_steps:int=-1):
        """
        Arguments:
            step_size {float} -- Step size of the agent. Defaults to 0.2.
            glob {bool} -- Whether to sample starting positions across the entire space. Defaults to False.
            freeze_agent {bool} -- Whether to freeze the agent's position until goal positions are hidden. Defaults to False.
        """
        self.freeze = freeze
        self._step_size = step_size
        self.max_episode_steps = max_episode_steps
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1    
        self._num_show_steps = 2    # this should determine for how many steps the goal is visible
        
        # Create an array with possible positions
        # Valid local positions are one tick away from 0.0 or between -0.4 and 0.4
        # Valid global positions are between -1 + step_size and 1 - step_size
        # Clipping has to be applied because step_size is a variable now
        num_steps = int( 0.4 / self._step_size)
        lower = min(- 2.0 * self._step_size, -num_steps * self._step_size) if not glob else -1  + self._step_size
        upper = max( 3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size) if not glob else 1

        self.possible_positions = np.arange(lower, upper, self._step_size).clip(-1 + self._step_size, 1 - self._step_size)
        self.possible_positions = list(map(lambda x: round(x, 2), self.possible_positions)) # fix floating point errors

        self.op = None


    def reset(self, **kwargs):
        """Resets the agent to a random start position and spawns the two possible goals randomly."""
        # Sample a random start position
        self._position = np.random.choice(self.possible_positions)
        self._rewards = []
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0])
        # Determine the goal
        self._goals = goals[np.random.permutation(2)]
        obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        return obs

    @property
    def observation_space(self):
        """
        Returns:
            {spaces.Box}: The agent observes its current position and the goal locations, which are masked eventually.
        """
        return spaces.Box(low = 0, high = 1.0, shape = (3,), dtype = np.float32)

    @property
    def action_space(self):
        """
        Returns:
            {spaces.Discrete}: The agent has two actions: going left or going right
        """
        return spaces.Discrete(2)

    def step(self, action):
        """
        Executes the agents action in the environment if the agent is allowed to move.

        Arguments:
            action {list} -- The agent action which should be executed.

        Returns:
            {numpy.ndarray} -- Observation of the agent.
            {float} -- Reward for the agent.
            {bool} -- Done flag whether the episode has terminated.
            {dict} -- Information about episode reward, length and agents success reaching the goal position
        """
        reward = 0.0
        done = False
        info = None
        success = False
        action = action[0]
        
        if self.max_episode_steps > 0 and self._step_count >= self.max_episode_steps - 1:
            done = True

        if self._num_show_steps > self._step_count:
            # Execute the agent action if agent is allowed to move
            self._position += self._step_size * (1 - self.freeze) if action == 1 else -self._step_size * (1 - self.freeze)
            self._position = np.round(self._position, 2)

            obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)

            if self.freeze: # Check if agent is allowed to move
                self._step_count += 1
                self._rewards.append(reward)
                return obs, reward, done, info

        else:
            self._position += self._step_size if action == 1 else -self._step_size
            self._position = np.round(self._position, 2)
            obs = np.asarray([0.0, self._position, 0.0], dtype=np.float32) # mask out goal information

        # Determine the reward function and episode termination
        if self._position == -1.0:
            if self._goals[0] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
                success = True
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        elif self._position == 1.0:
            if self._goals[1] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
                success = True
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        else:
            reward -= self._time_penalty
        self._rewards.append(reward)

        # Wrap up episode information
        if done:
            info = {"success": success,
                    "reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        # Increase step count
        self._step_count += 1

        return obs, reward, done, info

    def render(self):
        """
        A simple console render method for the environment.
        """
        if self.op is None:
            self.init_render = False
            self.op = output()
            self.op = self.op.warped_obj
            os.system('cls||clear')

            for _ in range(6):
                self.op.append("#")


        num_grids = 2 * int(1 / self._step_size) + 1
        agent_grid = int(num_grids / 2 + self._position / self._step_size) + 1

        self.op[1] = ('######' * num_grids +  "#")
        self.op[2] = ('#     ' * num_grids + "#")
        field = [*('#     ' * agent_grid)[:-3], *"a  ", *('#     ' * (num_grids - agent_grid)), "#"]

        if field[3] != "a":
            field[3] = "+" if self._goals[0] > 0 else "-"
        if field[-4] != "a":
            field[-4] = "+" if self._goals[1] > 0 else "-"

        self.op[3] = ("".join(field))
        self.op[4] = ('#     ' * num_grids + "#")
        self.op[5] = ('######' * num_grids + "#")
        
        self.op[6] = ("Goals are shown: " + str(self._num_show_steps > self._step_count))

        time.sleep(1.0) 

    def close(self):
        """
        Clears the used resources properly.
        """
        if self.op is not None:
            self.op.clear()
            self.op = None
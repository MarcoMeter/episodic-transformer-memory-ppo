from gym_minigrid.minigrid import *
from gym_minigrid.register import register

# Extend object IDs
OBJECT_TO_IDX["arrow_right"] = 11
OBJECT_TO_IDX["arrow_down"] = 12
OBJECT_TO_IDX["arrow_left"] = 13
OBJECT_TO_IDX["arrow_up"] = 14
OBJECT_TO_IDX["arrow_right_down"] = 15
OBJECT_TO_IDX["arrow_right_up"] = 16
OBJECT_TO_IDX["arrow_left_down"] = 17
OBJECT_TO_IDX["arrow_left_up"] = 18

COMMANDS = {
    "right"     : (1, 0),
    "down"      : (0, 1),
    "left"      : (-1, 0),
    "up"        : (0, -1),
    "stay"      : (0, 0),
    "right_down": (1, 1),
    "right_up"  : (1, -1),
    "left_down" : (-1, 1),
    "left_up"   : (-1, -1),
}

class ArrowRight(WorldObj):
    def __init__(self, color):
        super(ArrowRight, self).__init__("arrow_right", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*0.0)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowRightDown(WorldObj):
    def __init__(self, color):
        super(ArrowRightDown, self).__init__("arrow_right_down", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*0.5)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowRightUp(WorldObj):
    def __init__(self, color):
        super(ArrowRightUp, self).__init__("arrow_right_up", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*3.5)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowDown(WorldObj):
    def __init__(self, color):
        super(ArrowDown, self).__init__("arrow_down", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*1)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowLeft(WorldObj):
    def __init__(self, color):
        super(ArrowLeft, self).__init__("arrow_left", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*2)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowLeftDown(WorldObj):
    def __init__(self, color):
        super(ArrowLeftDown, self).__init__("arrow_left_down", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*1.5)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowLeftUp(WorldObj):
    def __init__(self, color):
        super(ArrowLeftUp, self).__init__("arrow_left_up", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*2.5)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowUp(WorldObj):
    def __init__(self, color):
        super(ArrowUp, self).__init__("arrow_up", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*3)
        fill_coords(img, tri_fn, COLORS[self.color])

class MortarEnv(MiniGridEnv):
    def __init__(self, seed=None, num_available_c=5, num_c=9, c_duration=5, c_delay=1, c_show_duration=1, show_all=False):
        if show_all:
            assert num_c <= 18, "Only 18 commands can be shown at once"
        assert num_available_c <= 9, "Only 9 commands are available"
        self.num_available_commands = num_available_c   # right, down, left, up, stay
        self.num_commands = num_c                       # How many commands to execute
        self.command_duration = c_duration              # How much time the agent has to move to the commanded position
        self.command_delay = c_delay                    # How many steps the next command is being delayed (i.e. no command is shown)
        self.command_show_duration = c_show_duration    # How many steps to show one command
        self.show_all_commands = show_all               # Whether to show one command at a time or all at once
        max_steps = self.num_commands * self.command_duration + 1
        if not self.show_all_commands:
            max_steps = max_steps + self.num_commands * (self.command_delay + self.command_show_duration)

        super().__init__(grid_size=9, max_steps=max_steps, seed=seed, see_through_walls=True)

    def _gen_grid(self, width, height):
        """_gen_grid is called every reset(). It generates a new grid and a news set of to be executed commands."""
        self._current_command = 0
        self._step_count_commands = 0
        self._commands_shown = False

        # Mission description
        self.mission = "execute the green commands correctly"

        # Create an empty grid and its surrounding walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(1, 1, width-2, height-2)

        # Sample a start position and rotation
        self.agent_pos = self._rand_pos(2, width - 2, 2, height - 2)
        self.agent_dir = self._rand_int(0, 3)

        # Sample n commands
        self._commands = self._generate_commands(self.agent_pos)

        if self.show_all_commands:
            # Render (place) commands in the first row of tiles
            for i in range(len(self._commands)):
                y = 0 if i < 9 else height - 1
                self._place_single_command(self._commands[i], pos=(i % 9, y))
        else:
            # Generate a list of what to be rendered during each step
            # This logic is more manageable during step()
            self._command_vis = self._generate_command_vis(self._commands, 1, 1)
            # Show the first command on the first observation
            self._place_single_command(self._command_vis[0], pos=(4, 0))
            self._command_vis.pop(0)

        # Initial target position
        # The target position is used to check whether the agent executed the command correctly
        self._target_pos = (self.agent_pos[0] + COMMANDS[self._commands[0]][0],
                            self.agent_pos[1] + COMMANDS[self._commands[0]][1])

    def _generate_command_vis(self, commands, duration=1, delay=0):
        """Generates a list that states on which step to show which command. Each element corresponds to one step.

        Args:
            commands {list} -- Sampled commands
            duration {int} -- How many steps to show one command (default: {1})
            delay {int} -- How many steps until the next command is shown (default: {0})

        Returns:
            {list} -- list that states on which step to show which command
        """
        command_vis = []
        for i in range(len(commands)):
            # Duplicate the command related to the duration
            for j in range(duration):
                command_vis.append(commands[i])
            # For each step delay, add None instead of the command
            for k in range(delay):
                command_vis.append(None)
        return command_vis

    def _generate_commands(self, start_pos):
        """Given the agent's start position generate a set of commands.

        Args:
            start_pos {truple} -- A tuple that indicates the initial x and y position of the agent

        Returns:
            {list} -- n sampled commands
        """
        simulated_pos = start_pos
        commands = []
        for i in range(self.num_commands):
            # Retrieve valid commands (we cannot walk on to a wall)
            valid_commands = self._get_valid_commands(simulated_pos)            
            # Sample one command from the available ones
            sample = self._rand_elem(valid_commands)
            commands.append(sample)
            # Update the simulated position
            simulated_pos = (simulated_pos[0] + COMMANDS[sample][0], simulated_pos[1] + COMMANDS[sample][1])
        return commands

    def _get_valid_commands(self, pos):
        """Given a position, which tile can the agent move to next? Actions are illegal if the agent cannot move to that position.

        Args:
            pos {tuple} -- A tuple that indicates the x and y position

        Returns:
            {list} -- A list of valid commands for the provided position
        """
        # Check whether each command can be executed or not
        valid_commands = []
        keys = list(COMMANDS.keys())[:self.num_available_commands]
        available_commands = {key: COMMANDS[key] for key in keys}
        for key, value in available_commands.items():
            obj = self.grid.get(pos[0] + value[0], pos[1] + value[1])
            if obj is None:
                valid_commands.append(key)
        # Return the commands that can be executed
        return valid_commands

    def _toggle_lava_on(self, target_pos):
        """Spawn lava on each tile, except for the target one

        Arguments:
            target_pos {tuple} -- A tuple that indicates the initial x and y position where lave shall not be placed
        """
        for i in range(2, 7):
            for j in range(2, 7):
                if target_pos != (i , j):
                    self.grid.set(i, j, Lava())
    
    def _toggle_lave_off(self):
        """Clears the entire lava"""
        for i in range(2, 7):
            for j in range(2, 7):
                self.grid.set(i, j, None)

    def _place_single_command(self, command, pos):
        """Places a command tile on the desired position.

        Arguments:
            command {str} -- The desired command
            pos {tuple} -- The desired position on the grid
        """
        obj = None
        if command == "right":
            obj = ArrowRight("green")
        elif command == "down":
            obj = ArrowDown("green")
        elif command == "left":
            obj = ArrowLeft("green")
        elif command == "up":
            obj = ArrowUp("green")
        elif command == "stay":
            obj = Ball("green")
        if command == "right_up":
            obj = ArrowRightUp("green")
        elif command == "right_down":
            obj = ArrowRightDown("green")
        elif command == "left_down":
            obj = ArrowLeftDown("green")
        elif command == "left_up":
            obj = ArrowLeftUp("green")
        self.grid.set(pos[0], pos[1], obj)
        return

    def step(self, action):
        # Case: All commands are shown at once during the entire episode
        if self.show_all_commands:
            self._commands_shown = True
        else:
            # Case: Commands were shown one by one and there is no remaining one
            if not self._command_vis:
                self._commands_shown = True
                self.grid.set(4, 0, None)

        # Case: Show each command one by one
        if not self._commands_shown:
            # Show commands
            # Determine and place the current command
            self._place_single_command(self._command_vis[0], pos=(4, 0))
            # Pop the shown command
            self._command_vis.pop(0)
            obs, reward, done, info = MiniGridEnv.step(self, MiniGridEnv.Actions.drop)
        # Case: Agent shall execute the commands. The agent is not frozen anymore.
        else:
            obs, reward, done, info = MiniGridEnv.step(self, action[0])
            # Process the command execution logic
            # One command is alive for command_duration steps
            if (self._step_count_commands) % (self.command_duration) == 0 and self._step_count_commands > 0:
                # Check if to be executed commands are still remaining
                if self._current_command < self.num_commands:
                    self._current_command += 1
                    # Toggle lava on for the next step
                    self._toggle_lava_on(self._target_pos)
                    # Check if the agent is on the target position
                    if tuple(self.agent_pos) == self._target_pos:
                        # Success!
                        if self._current_command < self.num_commands:
                            # Update target position
                            self._target_pos = (self._target_pos[0] + COMMANDS[self._commands[self._current_command]][0],
                                                self._target_pos[1] + COMMANDS[self._commands[self._current_command]][1])
                        reward = 0.1
                    # If the agent is not on the target position, terminate the episode
                    else:
                        # Failure!
                        done = True
                        reward = 0.0#-0.1
                # Finish the episode once all commands are completed
                if self._current_command >= self.num_commands:
                    # All commands completed!
                    done = True
            else:
                # Turn off lava
                self._toggle_lave_off()

            # Make sure that the lava signals a negative reward
            if done and reward == 0:
                reward = 0.0#-0.1
            
            # We cannot make use of the global step count due to potentially using the single command visualization
            self._step_count_commands +=1

        return obs, reward, done, info

class MortarABEnv(MortarEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, num_available_c=5, num_c=10, c_duration=5, c_delay=6, c_show_duration=1, show_all=False)

class MortarBEnv(MortarEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, num_available_c=5, num_c=10, c_duration=5, show_all=True)

register(
    id="MiniGrid-MortarAB-v0",
    entry_point="environments.mortar_env:MortarABEnv"
)

register(
    id="MiniGrid-MortarB-v0",
    entry_point="environments.mortar_env:MortarBEnv"
)
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt
import random
from typing import TYPE_CHECKING, List, Optional
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from anglr import Angle

MAX_XY_VEL = 4.0  # Maximum linear velocity (m/s)
MAX_ANG_VEL = 2.0  # Maximum angular velocity (rad/s)
MAX_X = np.float32(10.0)  # Maximum x position
MAX_Y = np.float32(10.0)  # Maximum y position
MIN_X = np.float32(-1.0) 
MIN_Y = np.float32(-1.0)
CLOSE_ENOUGH_REWARD = 30

class SwerveEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, render_mode: Optional[str] = None):
        super(SwerveEnv, self).__init__()

        self.render_mode = render_mode
        # Define action space: linear accelerations (X, Y) and angular acceleration (Z)
        self.action_space = spaces.Box(
            low=np.array(list(map(np.float32, [-1.0, -1.0, -1.0]))),
            high=np.array(list(map(np.float32, [1.0, 1.0, 1.0]))),
            dtype=np.float32
        )

        # Define observation space: position (x, y), orientation (theta), velocity (vx, vy, omega) target location (x, y, theta)
        self.observation_space = spaces.Box(
            low=np.array(list(map(np.float32, [MIN_X, MIN_Y, -np.pi, -MAX_XY_VEL, -MAX_XY_VEL, -MAX_ANG_VEL, MIN_X, MIN_Y, -np.pi]))),
            high=np.array(list(map(np.float32, [MAX_X, MAX_Y, np.pi, MAX_XY_VEL, MAX_XY_VEL, MAX_ANG_VEL, MAX_X, MAX_Y, np.pi]))),
            dtype=np.float32
        )


        # Time step size
        self.dt = 0.01  # seconds
        #self._target_location = np.array([random.uniform(1,9), random.uniform(1,9), Angle(random.uniform(-math.pi, math.pi))]) # Goal position
        self._target_location = np.array([9, 9, Angle(90, mode="degrees")]) # Goal position
        self.fig, self.ax = None, None
        self.last_reward = 0

        # Initialize state variables
        self.reset()

    def set_position(self, dx, dy, dtheta):
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta

    def _get_reward(self):
        if self.previous_distance is None:
            self.previous_distance = np.linalg.norm([self.state[0], self.state[1]] - self._target_location[:2])
            self.previous_angle_delta = -self._target_location[2].angle_between(Angle(self.state[2])).radians
            return 0
        angle_delta = -abs(self._target_location[2].angle_between(Angle(self.state[2])).radians)
        #print(f"angl: {angle_delta} previous_angle: {self.previous_angle_delta}")
        # Calculate reward (e.g., based on distance to a goal)
        #print(f"state: {self.state[0]} {self.state[1]} target: {self._target_location[0]} {self._target_location[1]}")
        distance = -math.sqrt((self.state[0] - self._target_location[0])**2 + (self.state[1] - self._target_location[1])**2)
        #print(f"distance: {distance} ")
        #print(f"distance: {distance} previous_distance: {self.previous_distance}")
        if abs(distance) < 0.1:
            #print("huge reward found")
            return CLOSE_ENOUGH_REWARD 
    
        return distance + angle_delta + 0.5 * self.state[3] + 0.5 * self.state[4] - 0.4 
        #return distance + angle_delta - 0.1  


    def _get_info(self):
        return {
            "reward": self._get_reward(),
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)        # Reset state: position (x, y), orientation (theta), and velocities (vx, vy, omega) target location (x, y, theta)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.close_enough_steps = 0 
        self.previous_distance = None
        self.previous_angle_delta = None
        self.num_steps = 0 

        return self.state, {}

    def step(self, action):
        self.num_steps += 1
        #print(f"num_steps: {self.num_steps}")
        # Unpack current state
        x, y, theta, vx, vy, omega, _,_,_ = self.state

        # Unpack action
        ax, ay, a_omega = action

        # Update velocities with accelerations
        # DOES NOT PRESERVE SIGN PROBLEMS WHEN GOING NEGATIVE
        # assumes 20 m/s^2 acceleration which is maybe fine? 
        vx += min(20 * ax * self.dt, MAX_XY_VEL)
        vy += min(20 * ay * self.dt, MAX_XY_VEL)
        omega += min(10 * a_omega * self.dt, MAX_ANG_VEL)

        # Update positions and orientation with velocities
        x += vx * self.dt
        y += vy * self.dt
        theta += omega * self.dt

        # Normalize theta to [-pi, pi]0.5
        self.state = np.array([x, y, theta, vx, vy, omega, self._target_location[0], self._target_location[1], self._target_location[2]], dtype=np.float32)
        reward = self._get_reward()
        self.last_reward = reward 
        # Define termination condition (e.g., timeout or out-of-bounds)
        terminated = False
        truncated = False 
        if reward == CLOSE_ENOUGH_REWARD:
            #print("Goal reached!")
            reward = 1000
            terminated = True
        if x > MAX_X or y > MAX_Y or x < MIN_X or y < MIN_Y:
            truncated = True
            #print("out of bounds")
            reward = -100 
        if self.num_steps > 300:
            truncated = True
            reward = -100
            #print("timeout")

        # Additional info (optional)
        info = {}

        return self.state, reward, terminated, truncated, info

    def _base_render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.set_aspect('equal')
            self.ax.grid(True)

        self.ax.clear()
        self.ax.set_xlim(MIN_X, MAX_X)
        self.ax.set_ylim(MIN_Y, MAX_Y)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        x, y, theta, _, _, _, _, _, _= self.state

        # Draw the swerve drive robot as a 25x25 square
        square = np.array([
            [-12.5, -12.5],
            [12.5, -12.5],
            [12.5, 12.5],
            [-12.5, 12.5],
            [-12.5, -12.5]
        ]) / 25.0  # Scale to normalized units

        # Rotate and translate square
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        transformed_square = (rotation_matrix @ square.T).T + np.array([x, y])

        self.ax.plot(transformed_square[:, 0], transformed_square[:, 1], 'b')
        self.ax.scatter(x, y, c='r', label="Position")
        # plot the goal point
        self.ax.scatter(self._target_location[0], self._target_location[1], c='g', label="Goal")
        # plot the target angle at the goal point (target location[2])
        self.ax.quiver(self._target_location[0], self._target_location[1], np.cos(self._target_location[2].radians), np.sin(self._target_location[2].radians), color='g', label="Goal Angle", width=0.003)
        # plot the current angle of the robot
        self.ax.quiver(x, y, np.cos(theta), np.sin(theta), color='r', label="Robot Angle", width=0.003)
        # put the reward in the title
        #self.ax.set_title(f"Reward: {self.last_reward}")
        # set also the distance to the goal in the title
        self.ax.set_title(f"Reward: {self.last_reward} \n Distance to goal: {-math.sqrt((self.state[0] - self._target_location[0])**2 + (self.state[1] - self._target_location[1])**2)}")
        self.ax.legend()   

    def _rgb_render(self):
        #print("rgb rendering")
        # Render the figure to an RGB array
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        rgb_image = np.frombuffer(canvas.tostring_argb(), dtype='uint8').reshape(height, width, 4)[:,:,1:]
        
        return rgb_image
    
    def _human_render(self):
        plt.pause(self.dt)

    def render(self):
        self._base_render()
        if self.render_mode == "human":
            self._human_render()
        elif self.render_mode == "rgb_array":
            return self._rgb_render()
        else:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make_vec("{self.spec.id}", render_mode="rgb_array")'
            )
            return

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig, self.ax = None, None


if __name__ == "__main__":
    import time
    env = SwerveEnv()
    env.reset()
    while True:
        env.render()
        env.set_position(0.01, 0.01, -0.01)
        time.sleep(0.1)

"""
Classic Van der Pol oscillator system  implemented by:
Alessandro Antonucci @AlexRookie
University of Trento
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class OscillatorVanDerPolEnv(gym.Env):
    """
    Description:
        The Van der Pol oscillator is a non-conservative oscillator with
        non-linear damping. It evolves in time according to a second-order
        differential equation, where the scalar parameter μ indicates the
        nonlinearity and the strength of the damping.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, dt=0.05, mu=2.0, force=[-1.0,1.0], x0=[0,1], v0=[-1.0,1.0], method="euler"):
        self.max_speed = 3
        self.min_force = force[0]
        self.max_force = force[1]
        self.dt = dt
        self.mu = mu
        self.x0 = x0
        self.v0 = v0
        self.kinematics_integrator = method
        self.state = None

        # Maximum oscillation
        self.x_threshold = 4.

        high = np.array([self.x_threshold, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(low=self.min_force, high=self.max_force, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_steps(self, seconds=10):
        return round(seconds/self.dt)

    def step(self, u):
        x, x_dot = self.state

        mu = self.mu
        dt = self.dt

        if isinstance(u, float):
            u = np.asarray([u], dtype=np.float32)
        u = np.clip(u, self.min_force, self.max_force)[0]
        self.last_u = u  # for rendering

        reward = 0.0
        #costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # TODO: metti reward

        xacc = u + mu * x_dot - mu * x_dot * np.power(x,2) - x

        if self.kinematics_integrator == "euler":
            # forward Euler
            new_x = x + dt * x_dot
            new_x_dot = x_dot + dt * xacc
        elif self.kinematics_integrator == "implicit":
            # semi-implicit Euler
            new_x_dot = x_dot + dt * xacc
            new_x = x + dt * new_x_dot

        self.state = np.array((new_x, new_x_dot), dtype=np.float32)

        done = bool(
            new_x < -self.x_threshold
            or new_x > self.x_threshold
        )

        return self.state, reward, done, {}

    def reset(self):
        state_x = self.np_random.uniform(low=self.x0[0], high=self.x0[1])
        state_v = self.np_random.uniform(low=self.v0[0], high=self.v0[1])
        self.state = np.array((state_x, state_v), dtype=np.float32)
        self.last_u = None
        return self.state

    def render(self, mode="human"):
        return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

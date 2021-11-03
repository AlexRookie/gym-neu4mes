"""
Classic linear oscillator system (mass-spring-damper model) implemented by:
Alessandro Antonucci @AlexRookie
University of Trento
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class OscillatorEnv(gym.Env):
    """
    Description:
        A linear oscillator can be regarded as one of the simplest system
        to model: it is formed by a mass connected to a spring and a damper,
        wichi can move with an oscillating behaviour along a single
        direction constrained by the spring and the damper.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, dt=0.05, m=1.0, c=0.175, k=3.0, force=[-1.0,1.0], x0=[0,1], v0=[-1.0,1.0], method="euler"):
        self.max_speed = 3
        self.min_force = force[0]
        self.max_force = force[1]
        self.tau = dt # seconds between state updates
        self.m = m
        self.c = c
        self.k = k
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

    def step(self, u):
        x, x_dot = self.state

        m = self.m
        c = self.c
        k = self.k
        dt = self.tau

        if isinstance(u, float):
            u = np.asarray([u], dtype=np.float32)
        u = np.clip(u, self.min_force, self.max_force)[0]
        self.last_u = u  # for rendering

        reward = 0.0
        #costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # TODO: metti reward

        xacc = (1 / m) * (u - c * x_dot - k * x)

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
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold
        scale = screen_width / world_width
        mass_y = screen_height / 2  # TOP OF MASS
        mass_width = 50.0
        mass_height = 30.0

        if self.viewer is None:
            from gym_neu4mes.envs import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #self.track = rendering.Line((0, mass_y), (screen_width, mass_y))
            #self.track.set_color(0, 0, 0)
            #self.viewer.add_geom(self.track)
            self.track = rendering.Line((screen_width / 2.0, screen_height / 2 - 50), (screen_width / 2.0, screen_height / 2 + 50), linewidth=3)
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            l, r, t, b = -mass_width / 2, mass_width / 2, mass_height / 2, -mass_height / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.set_color(0.8, 0.3, 0.3)
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            fname = path.join(path.dirname(__file__), "assets/left.png")
            self.img = rendering.Image(fname, 50.0, 50.0)
            self.img_trans = rendering.Transform()
            self.img.add_attr(self.img_trans)
        
        self.viewer.add_onetime(self.img)
        
        x = self.state
        mass_x = x[0] * scale + screen_width / 2.0  # MIDDLE OF MASS
        self.cart_trans.set_translation(mass_x, mass_y)
        print(self.last_u)
        if self.last_u is not None:
            #self.img_trans.scale = (2, 2)
            if self.last_u>0:
                self.img_trans.set_translation(mass_x + mass_width/2, mass_y)
                self.img_trans.set_rotation(np.pi)
            elif self.last_u<0:
                self.img_trans.set_translation(mass_x - mass_width/2, mass_y)
                self.img_trans.set_rotation(-np.pi)
            else:
                self.img_trans.scale = (0, 0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

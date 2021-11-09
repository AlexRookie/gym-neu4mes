import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Classic control
# ----------------------------------------

register(
    id='Oscillator-v0',
    entry_point='gym_neu4mes.envs:OscillatorEnv',
)

register(
    id='OscillatorDuffing-v0',
    entry_point='gym_neu4mes.envs:OscillatorDuffingEnv',
    kwargs={'type' : 'normal'},
)

register(
    id='OscillatorDuffing-v1',
    entry_point='gym_neu4mes.envs:OscillatorDuffingEnv',
    kwargs={'type' : 'modified'},
)

register(
    id='OscillatorVanDerPol-v0',
    entry_point='gym_neu4mes.envs:OscillatorVanDerPolEnv',
)

# Classic Control Gym
# ----------------------------------------

register(
    id="CartPole-v2",
    entry_point="gym_neu4mes.envs:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="Pendulum-v2",
    entry_point="gym_neu4mes.envs:PendulumEnv",
    max_episode_steps=200,
)
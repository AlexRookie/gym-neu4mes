import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Classic
# ----------------------------------------

register(
    id='Oscillator-v0',
    entry_point='gym_neu4mes.envs:OscillatorEnv',
)

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
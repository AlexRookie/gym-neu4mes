# gym-Neu4mes

## API

```python
import gym 
env = gym.make('gym_neu4mes:Oscillator-v0')

# env is created, now we can use it: 
for episode in range(10): 
    obs = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)
```

## Installation

```bash
cd gym-neu4mes
pip install -e .
```
## Citation

```
@misc{}
```

## Release Notes

TBD.

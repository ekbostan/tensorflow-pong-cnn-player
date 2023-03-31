
import gym
from colabgymrender.recorder import Recorder

#env = gym.make("PongNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")
directory = './video'
env = Recorder(env, directory)

observation = env.reset()
terminal = False
while not terminal:
  action = env.action_space.sample()
  observation, reward, terminal, info = env.step(action)
env.play()

# Test_dqn My model win 21-5 
import sys
from colabgymrender.recorder import Recorder
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
USE_CUDA = torch.cuda.is_available()

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
directory = './video'
env = Recorder(env, directory)

num_frames = 1000000
batch_size = 32
gamma = 0.99

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

# specify the path of the testing model
#from google.colab import drive
#drive.mount('/content/drive')
#pthname = '/content/drive/My Drive/PongNoFrameskip-v4-model.pth'

model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load(pthname, map_location='cpu'))
model.eval()
if USE_CUDA:
    model = model.cuda()
    print("Using cuda")

env.seed(1)
state = env.reset()
done = False
games_won = 0

while not done:
    action = model.act(state, 0)
    state, reward, done, _ = env.step(action)
    if reward != 0:
        print(reward)
    if reward == 1:
        games_won += 1

print("Points Won: {}".format(games_won))
env.play()
try:
    sys.exit(0)
except:
    pass

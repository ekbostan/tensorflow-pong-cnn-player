# run_dqn (train your model)
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1200000
batch_size = 32
gamma = 0.99
record_idx = 10000

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)

# specify the path of the testing model
#from google.colab import drive
#drive.mount('/content/drive')
#pthname ="/content/drive/My Drive/model_pretrained.pth"
#pthname = '/content/drive/My Drive/PongNoFrameskip-v4-model.pth'
model.load_state_dict(torch.load(pthname, map_location='cpu'))
target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model.copy_from(model)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    print("Using cuda")
    
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
losses = []
all_rewards = []
episode_reward = 0
state = env.reset()

for frame_idx in range(1, num_frames + 1):
  
    if frame_idx % 5000 == 0:
        saved_path = "/content/drive/My Drive/" + env_id + "-model.pth"
        torch.save(model.state_dict(), saved_path)
    
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])

    if frame_idx % 50000 == 0:
        target_model.copy_from(model)

# set your model path
saved_path = "/content/drive/My Drive/" + env_id + "-model.pth"
# save your final model
torch.save(model.state_dict(), saved_path)



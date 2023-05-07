# tensorflow-pong-cnn-player
TensorFlow-based DQL(Deep Q-learning) player for the classic video game "Pong".

https://user-images.githubusercontent.com/114015851/236653856-70ea34d1-1e8f-4acf-b2fd-fa1a49de14be.mp4



### cnn_layers.py

`layers.py` contains an implementation of the Noisy Linear layer module used in the DQN paper by DeepMind. The module provides a linear layer with added noise to enable better exploration during training. The module also includes methods for resetting the noise parameters.

### Atari Game Environment Wrappers

The Atari game environment wrappers are based on the OpenAI Gym framework and provide a set of enhancements to make the Atari games easier to work with when using the DQN algorithm. The following environment wrappers are included:

- `NoopResetEnv`: This wrapper samples initial states by taking a random number of no-ops on reset. No-op is assumed to be action 0.
- `FireResetEnv`: This wrapper takes action on reset for environments that are fixed until firing.
- `EpisodicLifeEnv`: This wrapper makes end-of-life equal to end-of-episode, but only resets on true game over. Done by DeepMind for the DQN algorithm and others since it helps with value estimation.

### Requirements

- Python 3.6+
- PyTorch 1.7+
- OpenAI Gym 0.17.2+
- NumPy 1.19.5+

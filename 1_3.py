import gym
import torch
import torch.functional as F

from torch.autograd import Variable


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    print(env.action_space,
          env.reset())

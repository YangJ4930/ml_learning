import gym
from warnings import filterwarnings
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

MEMORY_CAPACITY = 200
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
TARGET_REPLACE_ITER = 100

env = gym.make('CartPole-v1', render_mode="human")
env = env.unwrapped

n_action = env.action_space.n
n_states = env.observation_space.shape[0]


class Q(nn.Module):
    def __init__(self, ):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value


class Experience(object):

    def __init__(self):
        # 建表和表的大小
        self.memory_counter = 0
        # n_state是S，*2是因为s*2，+2是因为有r和a
        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 2))

    def push(self, s, a, r, s_):
        # 按照从上到小的方式构建表，然后把值放在了表的最开始的地方
        # hstack是堆叠的意思
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        # [index, :]的意思是第index行的所有列
        self.memory[index, :] = transition

        self.memory_counter += 1

    def sample(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :n_states])
        b_a = torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -n_states:])
        return b_s, b_a, b_r, b_s_


def choose_action(x, q_module):
    x_data = np.array(x)
    x = torch.unsqueeze(torch.FloatTensor(x_data), 0)
    if np.random.uniform() < EPSILON:
        actions_value = q_module.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()[0]
    else:
        action = np.random.randint(0, n_action)
    return action


if __name__ == '__main__':
    Q_Net = Q()
    Q_Hat = Q()
    Q_Hat.load_state_dict(Q_Net.state_dict())
    Q_Hat.eval()
    experience = Experience()
    optimizer = optim.RMSprop(Q_Net.parameters())
    loss_func = nn.MSELoss()
    # 前面的步骤先跑，不做优化，后面得到一定的数据之后再进行优化
    for i_episode in range(400):
        s = env.reset()[0]
        ep_r = 0
        while True:
            env.render()
            a = choose_action(s, Q_Net)
            s_, r, terminated, truncated, info = env.step(a)
            x, _, theta, _ = s_
            # 计算距离
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            experience.push(s, a, r, s_)
            ep_r += r
            # 一旦缓存的高于memory里的，就开始优化
            if experience.memory_counter > MEMORY_CAPACITY:
                b_s, b_a, b_r, b_s_ = experience.sample()
                q_eval = Q_Net(b_s).gather(1, b_a)
                q_next = Q_Hat(b_s_).detach()
                q_target = b_r + q_next.max(1)[0].unsqueeze(1)
                # 计算损失函数
                loss = loss_func(q_eval, q_target)
                optimizer.zero_grad()
                # 方向传播
                loss.backward()
                # 开始优化
                optimizer.step()
            if terminated or truncated:
                print('Episode: ', i_episode, '| Total Reward: ', round(ep_r, 2))
                break
            s = s_
        if i_episode % TARGET_REPLACE_ITER == 0:
            Q_Hat.load_state_dict(Q_Net.state_dict())

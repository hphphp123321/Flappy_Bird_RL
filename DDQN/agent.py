import random
from collections import deque
import numpy as np
from sqlalchemy import Float
import torch
import torch.nn as nn
from net import Net
import torch.optim as optim
import time
import copy
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

class Agent():
    def __init__(self, action_set):
        self.gamma = 0.999
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.modelT = self.init_netWork()
        self.batch_size = 128
        self.memory = deque(maxlen=50000)
        self.greedy = 1
        self.action_set = action_set
        self.mse = nn.MSELoss()
        self.lr_rate = 3e-4
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr_rate)
        self.train_step = 0
        self.update_time = 500
        

    def get_state(self, state):
        """
        提取游戏state中我们需要的数据
        :param state: 游戏state
        :return: 返回提取好的数据
        """
        return_state = np.zeros((3,))
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        velocity = state['player_vel']
        return_state[0] = dist_to_pipe_horz
        return_state[1] = dist_to_pipe_bottom
        return_state[2] = velocity

        # return_state = np.zeros((8,))
        # player_y = state["player_y"]
        # player_vel = state["player_vel"]
        # next_pipe_dist_to_player = state["next_pipe_dist_to_player"]
        # next_pipe_top_y = state["next_pipe_top_y"]
        # next_pipe_bottom_y = state["next_pipe_bottom_y"]
        # next_next_pipe_dist_to_player = state["next_next_pipe_dist_to_player"]
        # next_next_pipe_top_y = state["next_next_pipe_top_y"]
        # next_next_pipe_bottom_y = state["next_next_pipe_bottom_y"]

        # return_state[0] = player_y
        # return_state[1] = player_vel
        # return_state[2] = next_pipe_dist_to_player
        # return_state[3] = next_pipe_top_y
        # return_state[4] = next_pipe_bottom_y
        # return_state[5] = next_next_pipe_dist_to_player
        # return_state[6] = next_next_pipe_top_y
        # return_state[7] = next_next_pipe_bottom_y
        return return_state

    def init_netWork(self):
        """
        构建模型
        :return:
        """
        model1 = Net(in_dim=3, out_dim=2)
        model2 = Net(in_dim=3, out_dim=2)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        return model1.to(device=self.device), model2.to(device=self.device)

    def train_model(self):
        if len(self.memory) < 2500:
            return

        # print(f"len of memory:{len(self.memory)}")
        train_sample = random.sample(self.memory, k=self.batch_size)
        train_states = []
        next_states = []

        for sample in train_sample:
            cur_state, action, r, next_state, done = sample
            # print(f"current_state:{cur_state}, action:{action}, r:{r}, next_state:{next_state}, done:{done}")
            next_states.append(next_state)
            train_states.append(cur_state)
        # 转成np数组
        next_states = np.array(next_states)
        train_states = np.array(train_states)

        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        train_states = torch.tensor(train_states, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        # 得到下一个state的q值
        next_states_q = self.modelT(next_states)
        # print(f"next:{next_states_q}")
        # 得到预测值
        old_state_q = self.model(train_states)

        state_q = old_state_q.clone()
        
        for index, sample in enumerate(train_sample):
            cur_state, action, r, next_state, done = sample
            # 计算Q现实
            if not done:
                state_q[index][action] = r + self.gamma * torch.max(next_states_q[index])
            else:
                state_q[index][action] = r

        loss = self.mse(old_state_q, state_q)
        # print(f"loss:{loss/self.batch_size}")
        loss.backward()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr_rate)
        self.optimizer.step()
        # print(self.model.state_dict())

        if self.train_step % self.update_time == 0:
            self.modelT.load_state_dict(self.model.state_dict())
        self.train_step += 1


    def add_memory(self, sample):
        self.memory.append(sample)

    def update_greedy(self):
        if self.greedy > 0.001:
            self.greedy *= 0.995

    def get_best_action(self, state):
        if random.random() < self.greedy:
            return random.randint(0, 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            state_q = self.model(state)
            action = torch.argmax(state_q)
            # if action == 0:
            #     print(action)
            return action

    def act(self, p, action):
        """
        执行动作
        :param p: 通过p来向游戏发出动作命令
        :param action: 动作
        :return: 奖励
        """
        r = int(p.act(self.action_set[action]))
        # print(r)
        if r == 0:
            reward = 0.1
        elif r == 1:
            reward = 1
        else:
            reward = -1
        return reward

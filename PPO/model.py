from numpy import dtype
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from memory import Memory
from collections import deque
from torch.distributions.categorical import Categorical
from math import ceil

class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

class MLPCritic(nn.Module):


    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

class ActorCritic(nn.Module):
    def __init__(self,
                 # actor net MLP attributes:
                 input_dim_actor,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # critic net MLP attributes:
                 input_dim_critic,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 output_dim_actor,
                 output_dim_critic,
                 # actor/critic/ shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # machine size for problems, no business with network
        self.device = device

        self.actor = MLPActor(num_mlp_layers_actor, input_dim_actor, hidden_dim_actor, output_dim_actor).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, input_dim_critic, hidden_dim_critic, output_dim_critic).to(device)

    def forward(self,
                state:torch.Tensor
                ):
        pi_value = self.actor(state)
        pi = F.softmax(pi_value, dim=-1)
        v = self.critic(state)
        return pi, v

class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 # actor net MLP attributes:
                 output_dim_actor,
                 input_dim_actor,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # critic net MLP attributes:
                 output_dim_critic,
                 input_dim_critic,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # ppo attributes
                 decay_flag,
                 decay_step_size,
                 decay_ratio,
                 vloss_coef,
                 ploss_coef,
                 entloss_coef,
                 batch_size,
                 # actor/critic/ shared attribute
                 device
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.decay_flag = decay_flag
        self.decay_step_size = decay_step_size
        self.decay_ratio = decay_ratio
        self.vloss_coef = vloss_coef
        self.ploss_coef = ploss_coef
        self.entloss_coef = entloss_coef
        self.batch_size = batch_size
        self.device = device


        self.policy = ActorCritic(
                                  input_dim_actor=input_dim_actor,
                                  input_dim_critic=input_dim_critic,
                                  output_dim_actor=output_dim_actor,
                                  output_dim_critic=output_dim_critic,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=self.device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.decay_step_size,
                                                         gamma=self.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    # evaluate the actions
    def eval_actions(self, p: torch.Tensor, actions):
        softmax_dist = Categorical(p)
        ret = softmax_dist.log_prob(actions).reshape(-1)
        entropy = softmax_dist.entropy().mean()
        return ret, entropy

    def select_action(self, state: torch.Tensor) -> tuple:
        p, _ = self.policy_old(state)
        dist = Categorical(p)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action), log_prob

    def update(self, memories:Memory)->tuple:

        vloss_coef = self.vloss_coef
        ploss_coef = self.ploss_coef
        entloss_coef = self.entloss_coef
        batch_size = self.batch_size

        rewards = deque()

        discounted_reward = 0
        for reward, is_done in zip(reversed(memories.rewards), reversed(memories.is_done)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        actions = torch.tensor(np.array(memories.actions), dtype=torch.float)
        states = torch.tensor(np.array(memories.states), dtype=torch.float)
        print(states[0:4])
        print(actions.shape)
        old_log_probs = torch.tensor(memories.log_probs, dtype=torch.float)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            self.optimizer.zero_grad()
            for iter in range(ceil(len(memories)/batch_size)):
            # for iter in range(len(memories)//batch_size):
                cur_states = states[iter*batch_size:(iter+1)*batch_size]
                cur_actions = actions[iter*batch_size:(iter+1)*batch_size]
                cur_rewards = rewards[iter*batch_size:(iter+1)*batch_size]
                cur_old_log_probs = old_log_probs[iter*batch_size:(iter+1)*batch_size]
                pis, vals = self.policy(state=cur_states)
                logprobs, ent_loss = self.eval_actions(p=pis, actions=cur_actions)
                ratios = torch.exp(logprobs - cur_old_log_probs.detach())
                advantages = cur_rewards - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(1), cur_rewards)
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss.backward()
                loss_sum += loss.item()
                vloss_sum += v_loss.item()
            # loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if self.decay_flag:
            self.scheduler.step()
        return loss_sum/self.k_epochs, vloss_sum/self.k_epochs

from model import PPO
import torch
from memory import Memory

class Agent:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 # actor net MLP attributes:
                 input_dim_actor,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # critic net MLP attributes:
                 input_dim_critic,
                 output_dim_actor,
                 output_dim_critic,
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
                 device,
                 # memory attribute
                 memory_maxlen,
                 ):

        self.brain = PPO(lr=lr, 
                         gamma=gamma, 
                         k_epochs=k_epochs, 
                         eps_clip=eps_clip, 
                         input_dim_actor=input_dim_actor, 
                         num_mlp_layers_actor=num_mlp_layers_actor, 
                         hidden_dim_actor=hidden_dim_actor, 
                         input_dim_critic=input_dim_critic, 
                         output_dim_actor=output_dim_actor, 
                         output_dim_critic=output_dim_critic, 
                         num_mlp_layers_critic=num_mlp_layers_critic, 
                         hidden_dim_critic=hidden_dim_critic,
                         decay_flag=decay_flag, 
                         decay_step_size=decay_step_size, 
                         decay_ratio=decay_ratio, 
                         vloss_coef=vloss_coef, 
                         ploss_coef=ploss_coef, 
                         entloss_coef=entloss_coef, 
                         batch_size=batch_size, 
                         device=device
                        )
        self.memories = Memory(memory_maxlen=memory_maxlen)

    def select_action(self, state: torch.Tensor) -> int:
        action, log_prob = self.brain.select_action(state=state)
        return action, log_prob

    def add_memory(self, state:torch.Tensor, log_prob:torch.Tensor, action:int, reward:int, is_done:bool) -> None:
        state = state.tolist()
        self.memories.append(state=state, log_prob=log_prob, reward=reward, action=action, is_done=is_done)

    def learn(self) -> tuple:
        loss, vloss = self.brain.update(memories=self.memories)
        self.memories.clear()
        return loss, vloss
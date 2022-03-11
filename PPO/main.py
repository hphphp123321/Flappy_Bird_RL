from ple.games import flappybird
from agent import Agent
from env import FlappyBirdEnv
from Params import configs
import os
import torch
import numpy as np
import random
import time
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# RANDOM_SEED = 39
# torch.manual_seed(RANDOM_SEED)
# torch.cuda.manual_seed_all(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)
# torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    flappy = FlappyBirdEnv()
    bird = Agent(lr=configs.lr,
                gamma=configs.gamma,
                k_epochs=configs.k_epochs,
                eps_clip=configs.eps_clip,
                input_dim_actor=configs.input_dim_actor,
                num_mlp_layers_actor=configs.num_mlp_layers_actor,
                hidden_dim_actor=configs.hidden_dim_actor,
                input_dim_critic=configs.input_dim_critic,
                output_dim_actor=configs.output_dim_actor,
                output_dim_critic=configs.output_dim_critic,
                num_mlp_layers_critic=configs.num_mlp_layers_critic,
                hidden_dim_critic=configs.hidden_dim_critic,
                decay_flag=configs.decay_flag,
                decay_step_size=configs.decay_step_size,
                decay_ratio=configs.decay_ratio,
                vloss_coef=configs.vloss_coef,
                ploss_coef=configs.ploss_coef,
                entloss_coef=configs.entloss_coef,
                batch_size=configs.batch_size,
                device=configs.device,
                memory_maxlen=configs.memory_maxlen)
    max_episodes = configs.max_episodes


    for episode in range(max_episodes):
        flappy.reset()
        while True:
            # time.sleep(0.01)
            state = flappy.get_state()
            action, log_prob = bird.select_action(state=state)
            reward, is_done = flappy.step(action=action)
            bird.add_memory(state=state, log_prob=log_prob, action=action, reward=reward, is_done=is_done)
            if bird.memories.is_full():
                loss, vloss=bird.learn()
            if is_done:
                score = flappy.score
                max_score = flappy.max_score
                print(f"step:{flappy.step_num}, max_step:{flappy.max_step}, episode:{episode}, score:{score}, max_score:{max_score}")
                flappy.reset()
                break
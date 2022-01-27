import numpy as np
import time
from collections import deque
from ple import PLE
from ple.games import FlappyBird
from agent import Agent
import torch
import random
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"



if __name__ == "__main__":
    # 训练次数
    epochs = 2000000
    # 实例化游戏对象
    game = FlappyBird()
    # 类似游戏的一个接口，可以为我们提供一些功能
    p = PLE(game, fps=30, display_screen=True)
    # 初始化
    p.init()
    # 实例化Agent，将动作集传进去
    agent = Agent(p.getActionSet())
    max_score = 0

    for epoch in range(epochs):
        # 重置游戏
        p.reset_game()
        # 获得状态
        state = agent.get_state(game.getGameState())
        if epoch > 50:
            agent.update_greedy()
        step = 0
        current_score = 0
        # 获得最佳动作
        action = int(agent.get_best_action(state))
        while True:
            # 然后执行动作获得奖励
            reward = agent.act(p, action)
            next_state = agent.get_state(game.getGameState())
            next_action = int(agent.get_best_action(next_state))
            if reward == 1:
                current_score += 1
            E = []
            E.append((state, action, reward, next_state, next_action, p.game_over()))
            agent.train_model(E=E)
            # print(f"score:{p.score()}")
            state = next_state
            action = next_action
            step += 1
            if p.game_over():
                # 获得当前分数
                # current_score = p.score()
                max_score = max(max_score, current_score)
                print('第%s次游戏，得分为: %s,最大得分为: %s' % (epoch, current_score, max_score))
                if current_score >= 400:
                    state = {'net':agent.model.state_dict(), 'optimizer':agent.optimizer.state_dict(), 'epoch':epoch}
                    torch.save(state, f'./DQL/model/{epoch}_{current_score}.pth')
                    # exit(0)
                break


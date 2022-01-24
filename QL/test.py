import numpy as np
from agent import Agent
from ple import PLE
from ple.games.flappybird import FlappyBird
import time

if __name__ == "__main__":
    # 实例化游戏对象
    game = FlappyBird()
    # 类似游戏的一个接口，可以为我们提供一些功能
    p = PLE(game, fps=30, display_screen=True)
    # 初始化
    p.init()
    # 实例化Agent，将动作集传进去
    agent = Agent(p.getActionSet())
    agent.q_table = np.load("./866.0.npy")
    # 重置游戏
    p.reset_game()
    state = agent.get_state(game.getGameState())
    while True:
        # 获得最佳动作
        action = agent.get_best_action(state)
        print(action)
        # 然后执行动作获得奖励
        agent.act(p, action)
        next_state = agent.get_state(game.getGameState())
        
        state = next_state
        current_score = p.score()
        if p.game_over():
            print(current_score)
            break
        time.sleep(0.03333)

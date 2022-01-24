import numpy as np
from agent import Agent
from ple import PLE
from ple.games.flappybird import FlappyBird
import time


if __name__ == "__main__":
    # 训练次数
    episodes = 2000
    # 实例化游戏对象
    game = FlappyBird()
    # 类似游戏的一个接口，可以为我们提供一些功能
    p = PLE(game, fps=30, display_screen=True)
    # 初始化
    p.init()
    # 实例化Agent，将动作集传进去
    agent = Agent(p.getActionSet())
    max_score = 0
	
    for episode in range(episodes):
        # 重置游戏
        p.reset_game()
        # 获得状态
        state = agent.get_state(game.getGameState())
        agent.update_greedy()
        while True:
            # 获得最佳动作
            action = agent.get_best_action(state, greedy=True)
            # print(action)
            # 然后执行动作获得奖励
            reward = agent.act(p, action)
            # 获得执行动作之后的状态
            next_state = agent.get_state(game.getGameState())
            # 更新q-table
            agent.update_q_table(state, action, next_state, reward)
            # 获得当前分数
            current_score = p.score()
            state = next_state
            # time.sleep(0.03333)
            if p.game_over():
                print(game.getGameState()["player_y"] - game.getGameState()["next_pipe_top_y"])
                max_score = max(current_score, max_score)
                print('Episodes: %s, Current score: %s, Max score: %s' % (episode, current_score, max_score))
                # 保存q-table
                if current_score == max_score and max_score > 150:
                    np.save("{}.npy".format(current_score), agent.q_table)
                break
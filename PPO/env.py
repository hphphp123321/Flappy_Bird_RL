import numpy as np
from ple import PLE
from ple.games import FlappyBird
import torch


class FlappyBirdEnv():
    def __init__(self) -> None:
        self.game = FlappyBird()
        self.p = PLE(self.game, fps=30, display_screen=True)
        self.p.init()
        # self.game.setRNG(24)
        self.action_set = self.p.getActionSet()
        self.step_num = 0
        self.max_step = 0
        self.score = 0
        self.max_score = 0

    def step(self, action:int) -> list:
        state = self.get_state()
        reward = self.act_get_reward(action=action)
        next_state = self.get_state()
        done = self.is_done
        return state, action, reward, next_state, done

    def get_state(self) -> torch.Tensor:
        # return_state = np.zeros((3,))
        state = self.game.getGameState()

        # dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        # dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        # velocity = state['player_vel']
        # return_state[0] = dist_to_pipe_horz
        # return_state[1] = dist_to_pipe_bottom
        # return_state[2] = velocity
        return_state = np.zeros((8,))
        player_y = state["player_y"]
        player_vel = state["player_vel"]
        next_pipe_dist_to_player = state["next_pipe_dist_to_player"]
        next_pipe_top_y = state["next_pipe_top_y"]
        next_pipe_bottom_y = state["next_pipe_bottom_y"]
        next_next_pipe_dist_to_player = state["next_next_pipe_dist_to_player"]
        next_next_pipe_top_y = state["next_next_pipe_top_y"]
        next_next_pipe_bottom_y = state["next_next_pipe_bottom_y"]

        return_state[0] = player_y
        return_state[1] = player_vel
        return_state[2] = next_pipe_dist_to_player
        return_state[3] = next_pipe_top_y
        return_state[4] = next_pipe_bottom_y
        return_state[5] = next_next_pipe_dist_to_player
        return_state[6] = next_next_pipe_top_y
        return_state[7] = next_next_pipe_bottom_y
        return torch.tensor(return_state, dtype=torch.float)/128 - 1

    def step(self, action:int) -> tuple:
        self.step_num += 1
        reward = self.act_get_reward(action=action)
        is_done = self.is_done()
        return reward, is_done

    def act_get_reward(self, action:int) -> int:
        r = self.p.act(self.action_set[action])
        # print(r)
        if r == 0:
            reward = 1 + self.step_num/50000
        elif r == 1:
            reward = 10 + self.score/5000
            self.score += 1
        else:
            reward = -10
        return reward

    def is_done(self) -> bool:
        if self.p.game_over():
            return True
        return False

    def reset(self) -> None:
        if self.score >= self.max_score:
            self.max_score = self.score
            self.max_step = self.step_num
        self.score = 0
        self.step_num = 0
        self.p.reset_game()
        
        
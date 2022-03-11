from collections import deque
import torch

class Memory():
    def __init__(self, memory_maxlen: int) -> None:
        self.maxlen = memory_maxlen
        self.states = deque()
        self.log_probs = deque()
        self.actions = deque()
        self.rewards = deque()
        self.is_done = deque()

    def clear(self) -> None:
        self.states.clear()
        self.log_probs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.is_done.clear()

    def append(self, state:list, log_prob:torch.Tensor, action:int, reward:int, is_done:bool) -> None:
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_done.append(is_done)

    def __len__(self) ->int:
        return self.states.__len__()

    def is_full(self) -> bool:
        if self.states.__len__() >= self.maxlen:
            return True
        return False

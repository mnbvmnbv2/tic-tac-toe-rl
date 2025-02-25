import time
from dataclasses import dataclass

import gymnasium
import numpy as np
import torch

from pvp.c_tictactoe_pvp import TicTacToePVPEnv


@dataclass
class Settings:
    batch_size: int = 1


class TicTacToeEnvPy:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._batch_size = settings.batch_size

        # 19 = 9*2 bits + 1 for current_player
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(19,), dtype=np.int16
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.render_mode = None

        self.game_states = np.zeros((self._batch_size, 19), dtype=np.int16)
        # For multi-agent rewards, shape = (batch_size, 2)
        self.rewards = np.zeros((self._batch_size, 2), dtype=np.int16)
        self.done = np.zeros((self._batch_size), dtype=np.int16)
        self.winners = np.zeros((self._batch_size), dtype=np.int16)
        self.infos = [{} for _ in range(self._batch_size)]

        self._env = TicTacToePVPEnv(
            self.game_states,
            self.rewards,
            self.done,
            self.winners,
        )

        self.metadata = {"render_modes": []}

    def reset_all(self):
        self._env.reset_all()
        return self.game_states, self.infos

    def reset(self, idx: int):
        self._env.reset(idx)
        return self.game_states, self.infos

    def step(self, action: np.ndarray):
        """
        action should be shape (batch_size,) or something broadcastable.
        returns (states, rewards, done, infos).
        Each row of `rewards` is [rX, rO].
        """
        self._env.step(action)
        return self.game_states, self.rewards, self.done, self.infos


def speed_test(dim=1, nn=False):
    env = TicTacToeEnvPy(Settings(batch_size=dim))
    if nn:
        model = MLP(dim)
        model.eval()

    pre_time = time.time()
    num_steps = 0
    env.reset_all()
    while time.time() - pre_time < 1:
        if nn:
            actions = (
                model(torch.tensor(env.game_states, dtype=torch.float32))
                .argmax(dim=1, keepdim=True)
                .squeeze()
                .numpy(force=True)
                .astype(np.int16)
            )
        else:
            actions = np.random.randint(9, size=(dim,), dtype=np.int16)
        env.step(actions)
        num_steps += 1
        # we have auto-reset in the env
    print(f"Ran {num_steps * dim} steps in 1 seconds")


def memory_test(dim=1, n_steps=5):
    state_buffer = np.zeros((n_steps, dim, 19), dtype=np.int16)
    reward_buffer = np.zeros((n_steps, dim, 2), dtype=np.int16)
    done_buffer = np.zeros((n_steps, dim), dtype=np.int16)
    action_buffer = np.zeros((n_steps, dim), dtype=np.int16)
    env = TicTacToeEnvPy(Settings(batch_size=dim))
    env.reset_all()
    for i in range(n_steps):
        actions = np.random.randint(9, size=(dim,), dtype=np.int16)
        env.step(actions)
        state_buffer[i] = env.game_states
        reward_buffer[i] = env.rewards
        done_buffer[i] = env.done
        action_buffer[i] = actions
    print(f"Ran {i + 1} steps with dim={dim}")

    print(f"state_buffer: {state_buffer.nbytes / 1024} KB")
    print(f"reward_buffer: {reward_buffer.nbytes / 1024} KB")
    print(f"done_buffer: {done_buffer.nbytes / 1024} KB")
    print(f"action_buffer: {action_buffer.nbytes / 1024} KB")

    print(f"State buffer: {state_buffer}")
    print(f"Reward buffer: {reward_buffer}")
    print(f"Done buffer: {done_buffer}")
    print(f"Action buffer: {action_buffer}")


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(18, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # for i in range(1, 101, 10):
    #     print(f"dim={i}")
    #     speed_test(i)
    speed_test(1000, nn=False)
    # memory_test(100, 100_000)  # 100_000)
    # print(torch.cuda.is_available())

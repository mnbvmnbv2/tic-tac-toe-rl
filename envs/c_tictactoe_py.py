import time
from dataclasses import dataclass

import gymnasium
import numpy as np

from c_tictactoe import TicTacToeEnv


@dataclass
class Settings:
    batch_size: int = 1


class TicTacToeEnvPy:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self._batch_size = settings.batch_size

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None

        self.game_states = np.zeros((self._batch_size, 18), dtype=np.int16)
        self.rewards = np.zeros((self._batch_size), dtype=np.int16)
        self.done = np.zeros((self._batch_size), dtype=np.int16)
        self.winners = np.zeros((self._batch_size), dtype=np.int16)
        self.infos = [{} for _ in range(self._batch_size)]

        self._env = TicTacToeEnv(
            self.game_states,
            self.rewards,
            self.done,
            self.winners,
        )

        self.metadata = {"render_modes": []}

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        self._env.reset()
        return self.game_states, self.infos

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self._env.step(action)
        return self.game_states, self.rewards, self.done, self.infos


def speed_test(dim=1):
    env = TicTacToeEnvPy(Settings(batch_size=dim))
    pre_time = time.time()
    num_steps = 0
    while time.time() - pre_time < 1:
        env.reset()
        while not env.done[0]:
            # print(done)
            # print(game_states)
            env.step(np.random.randint(9, size=(dim,), dtype=np.int16))
            num_steps += 1
    print(f"Ran {num_steps * dim} steps in 1 seconds")


if __name__ == "__main__":
    for i in range(1, 25):
        print(f"dim={i}")
        speed_test(i)

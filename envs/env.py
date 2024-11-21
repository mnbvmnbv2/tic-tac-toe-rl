import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

import gymnasium


class GameStateSetting(Enum):
    FLAT = 0
    GRID = 1


@dataclass
class Settings:
    game_state: GameStateSetting = GameStateSetting.FLAT


class TicTacToeEnv:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None

        if self._settings.game_state == GameStateSetting.FLAT:
            self.game_state = np.zeros((9), dtype=np.uint8)
        elif self._settings.game_state == GameStateSetting.GRID:
            self.game_state = np.zeros((3, 3), dtype=np.uint8)
        else:
            raise ValueError("Invalid game state setting")

        self.reward = np.zeros(1, dtype=np.float32)
        self.terminated = np.zeros(1, dtype=bool)
        self.truncated = np.zeros(1, dtype=bool)
        self.info = {}
        self.observation = np.zeros((18), dtype=np.uint8)

        self.metadata = {"render_modes": []}

    def close(self) -> None:
        pass

    def get_obs(self) -> np.ndarray:
        if self._settings.game_state == GameStateSetting.FLAT:
            return self._get_obs_flat()
        elif self._settings.game_state == GameStateSetting.GRID:
            return self._get_obs_3x3()
        else:
            raise ValueError("Invalid game state setting")

    def _get_obs_flat(self) -> np.ndarray:
        self.observation = np.eye(3)[self.game_state][:, 1:].flatten().astype(np.uint8)
        return self.observation

    def _get_obs_3x3(self) -> np.ndarray:
        self.observation = self.game_state.ravel().astype(np.uint8)
        return self.observation

    def _get_win_loop(self) -> int:
        # 0 for tie, 1 for player 1, 2 for player 2
        # rows
        for i in range(3):
            if (
                self.game_state[i * 3]
                == self.game_state[i * 3 + 1]
                == self.game_state[i * 3 + 2]
                != 0
            ):
                return self.game_state[i * 3]
        # columns
        for i in range(3):
            if (
                self.game_state[i]
                == self.game_state[i + 3]
                == self.game_state[i + 6]
                != 0
            ):
                return self.game_state[i]
        # diagonals
        if self.game_state[0] == self.game_state[4] == self.game_state[8] != 0:
            return self.game_state[0]
        if self.game_state[2] == self.game_state[4] == self.game_state[6] != 0:
            return self.game_state[2]
        return 0

    def _get_win_3x3(self) -> int:
        # 0 for tie, 1 for player 1, 2 for player 2
        # rows
        rows_equal = np.all(self.game_state == self.game_state[:, [0]], axis=1) & (
            self.game_state[:, 0] != 0
        )
        if np.any(rows_equal):
            return self.game_state[rows_equal][0, 0]
        # columns
        cols_equal = np.all(self.game_state == self.game_state[[0], :], axis=0) & (
            self.game_state[0, :] != 0
        )
        if np.any(cols_equal):
            return self.game_state[0, cols_equal][0]
        # diagonals
        if self.game_state[0, 0] == self.game_state[1, 1] == self.game_state[2, 2] != 0:
            return self.game_state[0]
        if self.game_state[2, 0] == self.game_state[1, 1] == self.game_state[0, 2] != 0:
            return self.game_state[2]
        return 0

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        if self._settings.game_state == GameStateSetting.FLAT:
            self.game_state = np.zeros(9, dtype=np.uint8)
        elif self._settings.game_state == GameStateSetting.GRID:
            self.game_state = np.zeros((3, 3), dtype=np.uint8)
        else:
            raise ValueError("Invalid game state setting")
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.get_obs()

        return self.get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._settings.game_state == GameStateSetting.FLAT:
            return self._step_flat(action)
        elif self._settings.game_state == GameStateSetting.GRID:
            return self._step_3x3(action)
        else:
            raise ValueError("Invalid game state setting")

    def _step_flat(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        # if illegal move
        if self.game_state[action] > 0:
            self.reward = -1
            self.terminated = False
            self.truncated = False
            return (
                self.observation,
                self.reward,
                self.terminated,
                self.truncated,
                self.info,
            )
        # else, player performs the action
        self.game_state[action] = 1

        # check if done (player 1 is always last in tied game)
        is_done = np.all(self.game_state > 0)
        winner = self._get_win_loop()
        if winner > 0 or is_done:
            # winner can only be player 1 else it's a tie
            self.reward = winner
            self.terminated = True
            self.truncated = False
            return (
                self.observation,
                self.reward,
                self.terminated,
                self.truncated,
                self.info,
            )

        # random move by the opponent
        opponent_action = np.where(self.game_state == 0)[0]
        opponent_action = np.random.choice(opponent_action)
        self.game_state[opponent_action] = 2
        winner = self._get_win_loop()
        if winner > 0:
            # only player 2 can win here
            self.reward = -1
            self.terminated = True
            self.truncated = False
            return (
                self.observation,
                self.reward,
                self.terminated,
                self.truncated,
                self.info,
            )
        # else we continue the game
        self.reward = 0
        self.terminated = False
        self.truncated = False
        return self.get_obs(), self.reward, self.terminated, self.truncated, self.info

    def _step_3x3(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        # turn int into (x, y)
        action = (action // 3, action % 3)
        # if illegal move
        if self.game_state[action] > 0:
            return self.get_obs(), -1, False, False, {}
        # else, player performs the action
        self.game_state[action] = 1

        # check if done (player 1 is always last in tied game)
        is_done = np.all(self.game_state > 0)
        winner = self._get_win_3x3()
        if winner > 0 or is_done:
            match winner:
                case 0:
                    reward = 0
                case 1:
                    reward = 1
                case 2:
                    reward = -1
            return self.get_obs(), reward, True, False, {}

        # random move by the opponent
        opponent_action = np.where(self.game_state == 0)
        opponent_action_1 = np.random.choice(opponent_action[0])
        opponent_action_2 = np.random.choice(opponent_action[1])
        self.game_state[opponent_action_1, opponent_action_2] = 2
        winner = self._get_win_3x3()
        if winner > 0:
            # only player 2 can win here
            return self.get_obs(), -1, True, False, {}
        # else we continue the game
        return self.get_obs(), 0, False, False, {}

    def nice_print(self) -> str:
        if self._settings.game_state == GameStateSetting.FLAT:
            output = "\n".join(
                [
                    " ".join(
                        [[" ", "X", "O"][x] for x in self.game_state[i * 3 : i * 3 + 3]]
                    )
                    for i in range(3)
                ]
            )
        elif self._settings.game_state == GameStateSetting.GRID:
            output = "\n".join(
                [
                    " ".join([[" ", "X", "O"][x] for x in self.game_state[i]])
                    for i in range(3)
                ]
            )
        else:
            raise ValueError("Invalid game state setting")
        return output


def test(settings: Settings):
    env = TicTacToeEnv(settings)
    obs, info = env.reset()
    print(env.nice_print())
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(
            int(input("Enter action: "))
        )
        print(env.nice_print())
        if terminated or truncated:
            print(f"Game ended with reward {reward}")
            break


def speed_test(settings: Settings):
    print(f"Running speed test with {settings}")
    env = TicTacToeEnv(settings)
    num_steps = 0
    rng = np.random.default_rng()
    env.reset()
    pre_time = time.time()
    while time.time() - pre_time < 1:
        action = rng.integers(0, 9)
        obs, reward, terminated, truncated, info = env.step(action)
        num_steps += 1
        # restart the terminated or truncated games
        if terminated or truncated:
            env.reset()

    print(f"Ran {num_steps} steps in 1 second")


if __name__ == "__main__":
    speed_test(Settings(game_state=GameStateSetting.GRID))
    speed_test(Settings(game_state=GameStateSetting.FLAT))
    # test(Settings(game_state=GameStateSetting.GRID))

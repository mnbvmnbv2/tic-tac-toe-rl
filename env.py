import time

import numpy as np

import gymnasium
import pufferlib


class TicTacToeEnvSingle(pufferlib.PufferEnv):
    def __init__(self) -> None:
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations=np.zeros((1, 18), dtype=np.uint8),
            rewards=np.zeros(1, dtype=np.float32),
            terminals=np.zeros(1, dtype=bool),
            truncations=np.zeros(1, dtype=bool),
            masks=np.ones(1, dtype=bool),
        )
        self.actions = np.zeros(1, dtype=np.uint32)

        self.game_state = np.zeros(9, dtype=np.uint8)

    def get_obs(self) -> np.ndarray:
        # flatten one-hot encoding
        return np.eye(3)[self.game_state][:, 1:].flatten()

    def check_win(self) -> int:
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

    def reset(self, seed=None) -> tuple[np.ndarray, dict]:
        # obs, info
        self.game_state = np.zeros(9, dtype=np.uint8)
        return self.get_obs(), {}

    def nice_print(self) -> str:
        return "\n".join(
            [
                " ".join(
                    [[" ", "X", "O"][x] for x in self.game_state[i * 3 : i * 3 + 3]]
                )
                for i in range(3)
            ]
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        # if illegal move
        if self.game_state[action] > 0:
            return self.get_obs(), -1, False, False, {}
        # else, player performs the action
        self.game_state[action] = 1

        # check if done (player 1 is always last in tied game)
        is_done = np.all(self.game_state > 0)
        winner = self.check_win()
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
        opponent_action = np.where(self.game_state == 0)[0]
        opponent_action = np.random.choice(opponent_action)
        self.game_state[opponent_action] = 2
        winner = self.check_win()
        if winner > 0:
            # only player 2 can win here
            return self.get_obs(), -1, True, False, {}
        # else we continue the game
        return self.get_obs(), 0, False, False, {}


def test():
    env = TicTacToeEnvSingle()
    obs, info = env.reset()
    print(obs.reshape((3, 6)))
    print(env.nice_print())
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(
            int(input("Enter action: "))
        )
        print(obs.reshape((3, 6)))
        print(env.nice_print())
        if terminated:
            print(f"Game ended with reward {reward}")
            break


def speed_test():
    env = TicTacToeEnvSingle()
    pre_time = time.time()
    num_steps = 0
    while time.time() - pre_time < 5:
        obs, info = env.reset()
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(np.random.randint(9))
            num_steps += 1
    print(f"Ran {num_steps} steps in 5 second")


if __name__ == "__main__":
    speed_test()

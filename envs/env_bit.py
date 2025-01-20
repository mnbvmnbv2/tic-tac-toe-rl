import time

import numpy as np

import gymnasium

"""
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0
-----------------------------------
18 bit for game staet
1 for reward
1 for done
nvm reward must be two bits for positive negative or 0 reward.

20 bits per env
"""


class TicTacToeEnv:
    def __init__(self) -> None:
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.render_mode = None

        self.state = 0b00000000000000000
        self.reward = 0b00
        self.terminated = 0b0
        self.truncated = 0b0
        self.info = {}
        self.observation = np.zeros((18), dtype=np.uint8)

        self.metadata = {"render_modes": []}

    def close(self) -> None:
        pass

    def get_obs(self) -> np.ndarray:
        self.observation = np.eye(3)[self.state][:, 1:].flatten().astype(np.uint8)
        return self.observation

    def _get_win(self) -> int:
        # 0 for tie, 1 for player 1, 2 for player 2
        for i in range(2):
            # rows
            for i in range(3):
                win_cond = 0b101010000000000000
                win_cond = win_cond >> i
                won = (win_cond & self.state) == win_cond
                if won:
                    return i
                win_cond = win_cond >> 6
            # columns
            for i in range(3):
                win_cond_col1 = 0b100000100000100000
                win_cond_col2 = 0b001000001000001000
                win_cond_col3 = 0b000010000010000010
                win_cond = win_cond >> i
                won = (win_cond & self.state) == win_cond
                if won:
                    return i
                win_cond = win_cond >> 6
            # diagonals
            if self.state[0] == self.state[4] == self.state[8] != 0:
                return self.state[0]
            if self.state[2] == self.state[4] == self.state[6] != 0:
                return self.state[2]
        return 0

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        self.state = np.zeros(9, dtype=np.uint8)
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.get_obs()

        return self.get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        # if illegal move
        if self.state[action] > 0:
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
        self.state[action] = 1

        # check if done (player 1 is always last in tied game)
        is_done = np.all(self.state > 0)
        winner = self._get_win()
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
        opponent_action = np.where(self.state == 0)[0]
        opponent_action = np.random.choice(opponent_action)
        self.state[opponent_action] = 2
        winner = self._get_win()
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

    def nice_print(self) -> str:
        output = "\n".join(
            [
                " ".join([[" ", "X", "O"][x] for x in self.state[i * 3 : i * 3 + 3]])
                for i in range(3)
            ]
        )
        return output


def speed_test():
    """Single env random policy steps per second test"""
    print("Running speed test")
    env = TicTacToeEnv()
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
    speed_test()

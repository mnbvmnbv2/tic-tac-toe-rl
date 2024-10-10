import time

import numpy as np

import gymnasium


class TicTacToeEnvSingle:
    def __init__(self, batch_size: int = 1) -> None:
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None

        self._batch_size = batch_size

        self.game_state = np.zeros((batch_size, 9), dtype=np.uint8)
        self.reward = np.zeros(self._batch_size, dtype=np.float32)
        self.terminated = np.zeros(self._batch_size, dtype=bool)
        self.truncated = np.zeros(self._batch_size, dtype=bool)
        self.info = [{} for _ in range(self._batch_size)]
        self.observation = np.zeros((self._batch_size, 18), dtype=np.uint8)

        self._winners = np.zeros(batch_size, dtype=np.uint8)

    def calc_obs(self) -> None:
        # flatten one-hot encoding
        self.observation = (
            np.eye(3)[self.game_state][:, :, 1:].flatten().reshape(self._batch_size, 18)
        )

    def check_win(self) -> None:
        # 0 for tie, 1 for player 1, 2 for player 2
        # rows
        for g in range(self._batch_size):
            for i in range(3):
                if (
                    self.game_state[g, i * 3]
                    == self.game_state[g, i * 3 + 1]
                    == self.game_state[g, i * 3 + 2]
                    != 0
                ):
                    self._winners[g] = self.game_state[g, i * 3]
            # columns
            for i in range(3):
                if (
                    self.game_state[g, i]
                    == self.game_state[g, i + 3]
                    == self.game_state[g, i + 6]
                    != 0
                ):
                    self._winners[g] = self.game_state[g, i]
            # diagonals
            if (
                self.game_state[g, 0]
                == self.game_state[g, 4]
                == self.game_state[g, 8]
                != 0
            ):
                self._winners[g] = self.game_state[g, 0]
            if (
                self.game_state[g, 2]
                == self.game_state[g, 4]
                == self.game_state[g, 6]
                != 0
            ):
                self._winners[g] = self.game_state[g, 2]
            self._winners[g] = 0

    def _reset(self, game_idx: int) -> None:
        self.game_state[game_idx] = np.zeros(9, dtype=np.uint8)
        self.reward[game_idx] = 0
        self.terminated[game_idx] = False
        self.truncated[game_idx] = False
        self.info[game_idx] = {}
        # :(
        self.calc_obs()

    def reset_all(self, seed=None) -> tuple[np.ndarray, dict]:
        # obs, info
        for g in range(self._batch_size):
            self._reset(g)

        return self.calc_obs(), {}

    def nice_print(self) -> str:
        return "\n".join(
            [
                " ".join(
                    [[" ", "X", "O"][x] for x in self.game_state[i * 3 : i * 3 + 3]]
                )
                for i in range(3)
            ]
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        for g in range(self._batch_size):
            # if illegal move
            if self.game_state[g, action[g]] > 0:
                self.reward[g] = -1
                self.terminated[g] = False
                self.truncated[g] = False
                continue
            # else, player performs the action
            self.game_state[g, action[g]] = 1

            # check if done (player 1 is always last in tied game)
            is_done = np.all(self.game_state[g] > 0)
            self.check_win()
            if self._winners[g] > 0 or is_done:
                match self._winners[g]:
                    case 0:
                        reward = 0
                    case 1:
                        reward = 1
                    case 2:
                        reward = -1
                self.reward[g] = reward
                self.terminated[g] = True
                self.truncated[g] = False
                continue

            # random move by the opponent
            opponent_action = np.where(self.game_state[g] == 0)[0]
            opponent_action = np.random.choice(opponent_action)
            self.game_state[g, opponent_action] = 2
            self.check_win()
            if self._winners[g] > 0:
                # only player 2 can win here
                self.reward[g] = -1
                self.terminated[g] = True
                self.truncated[g] = False
                continue
            # else we continue the game
            self.reward[g] = 0
            self.terminated[g] = False
            self.truncated[g] = False
        self.calc_obs()
        return self.observation, self.reward, self.terminated, self.truncated, self.info


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


def speed_test(dims=1):
    env = TicTacToeEnvSingle(batch_size=dims)
    pre_time = time.time()
    num_steps = 0
    rng = np.random.default_rng()
    env.reset_all()
    while time.time() - pre_time < 1:
        action = rng.integers(9, size=env._batch_size)
        obs, reward, terminated, truncated, info = env.step(action)
        num_steps += 1
        # restart the terminated or truncated games
        for g in range(dims):
            if terminated[g] or truncated[g]:
                env._reset(g)

    print(f"Ran {num_steps * dims} steps in 1 second")


if __name__ == "__main__":
    for i in range(10):
        speed_test(i)

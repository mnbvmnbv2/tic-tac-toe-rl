import time

import numpy as np

import gymnasium


class TicTacToeEnvSingle:
    def __init__(self) -> None:
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None

        self.game_state = np.zeros((9), dtype=np.uint8)
        self.reward = np.zeros(1, dtype=np.float32)
        self.terminated = np.zeros(1, dtype=bool)
        self.truncated = np.zeros(1, dtype=bool)
        self.info = {}
        self.observation = np.zeros((18), dtype=np.uint8)

        self._winners = np.zeros(1, dtype=np.uint8)

        self.metadata = {"render_modes": []}

    def calc_obs(self) -> np.ndarray:
        # flatten one-hot encoding
        self.observation = (
            np.eye(3)[self.game_state][:, :, 1:].flatten().astype(np.uint8)
            # .reshape(self._batch_size, 18)
        )
        print("a ", self.observation)

        return self.observation

    def check_win(self) -> None:
        # 0 for tie, 1 for player 1, 2 for player 2
        # rows
        for i in range(3):
            if (
                self.game_state[i * 3]
                == self.game_state[i * 3 + 1]
                == self.game_state[i * 3 + 2]
                != 0
            ):
                self._winners = self.game_state[i * 3]
        # columns
        for i in range(3):
            if (
                self.game_state[i]
                == self.game_state[i + 3]
                == self.game_state[i + 6]
                != 0
            ):
                self._winners = self.game_state[i]
        # diagonals
        if self.game_state[0] == self.game_state[4] == self.game_state[8] != 0:
            self._winners = self.game_state[0]
        if self.game_state[2] == self.game_state[4] == self.game_state[6] != 0:
            self._winners = self.game_state[2]
        self._winners = 0

    def _reset(self) -> None:
        self.game_state = np.zeros(9, dtype=np.uint8)
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        # :(
        self.calc_obs()

    def reset(
        self,
        seed,
        options,
    ) -> tuple[np.ndarray, dict]:
        self._reset()

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
        print("action", action)
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
        self.check_win()
        if self._winners > 0 or is_done:
            # winner can only be player 1 else it's a tie
            self.reward = self._winners
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
        self.check_win()
        if self._winners > 0:
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

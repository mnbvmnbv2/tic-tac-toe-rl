import time

import cupy as cp

import gymnasium


class TicTacToeEnv:
    def __init__(self, batch_size: int = 1) -> None:
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(18,), dtype=cp.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None

        self._batch_size = batch_size

        self.game_state = cp.zeros((batch_size, 9), dtype=cp.uint8)
        self.reward = cp.zeros(self._batch_size, dtype=cp.float32)
        self.terminated = cp.zeros(self._batch_size, dtype=bool)
        self.truncated = cp.zeros(self._batch_size, dtype=bool)
        self.info = [{} for _ in range(self._batch_size)]
        self.observation = cp.zeros((self._batch_size, 18), dtype=cp.uint8)

        self._winners = cp.zeros(batch_size, dtype=cp.uint8)

    def calc_obs(self) -> None:
        # flatten one-hot encoding
        self.observation = (
            cp.eye(3)[self.game_state][:, :, 1:].flatten().reshape(self._batch_size, 18)
        )

    def check_win(self) -> None:
        reshaped_state = self.game_state.reshape(self._batch_size, 3, 3)
        # 0 for tie, 1 for player 1, 2 for player 2
        # rows
        rows_equal = cp.all(reshaped_state == reshaped_state[:, :, [0]], axis=2) & (
            reshaped_state[:, :, 0] != 0
        )
        row_winners = reshaped_state[:, :, 0] * rows_equal
        cols_equal = cp.all(reshaped_state == reshaped_state[:, [0], :], axis=1) & (
            reshaped_state[:, 0, :] != 0
        )
        col_winners = reshaped_state[:, 0, :] * cols_equal
        diagonal_1 = (
            (reshaped_state[:, 0, 0] != 0)
            & (reshaped_state[:, 0, 0] == reshaped_state[:, 1, 1])
            & (reshaped_state[:, 0, 0] == reshaped_state[:, 2, 2])
        )
        diagonal_1_winners = reshaped_state[:, 0, 0] * diagonal_1
        diagonal_2 = (
            (reshaped_state[:, 0, 2] != 0)
            & (reshaped_state[:, 0, 2] == reshaped_state[:, 1, 1])
            & (reshaped_state[:, 0, 2] == reshaped_state[:, 2, 0])
        )
        diagonal_2_winners = reshaped_state[:, 0, 2] * diagonal_2

        self._winners = cp.maximum(
            row_winners.max(axis=1),
            cp.maximum(
                col_winners.max(axis=1),
                cp.maximum(diagonal_1_winners, diagonal_2_winners),
            ),
        )

    def _reset(self, game_idx: int) -> None:
        self.game_state[game_idx] = cp.zeros(9, dtype=cp.uint8)
        self.reward[game_idx] = 0
        self.terminated[game_idx] = False
        self.truncated[game_idx] = False
        self.info[game_idx] = {}
        # :(
        self.calc_obs()

    def reset_all(self, seed=None) -> tuple[cp.ndarray, dict]:
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

    def step(self, action: cp.ndarray) -> tuple[cp.ndarray, float, bool, bool, dict]:
        batch_size = self._batch_size
        indices = cp.arange(batch_size)

        # Initialize arrays for rewards and statuses
        self.reward = cp.zeros(batch_size, dtype=float)
        self.terminated = cp.zeros(batch_size, dtype=bool)
        self.truncated = cp.zeros(batch_size, dtype=bool)

        # Check for illegal moves
        illegal_move = self.game_state[indices, action] > 0
        self.reward[illegal_move] = -1
        self.terminated[illegal_move] = False
        self.truncated[illegal_move] = False

        # Mask for games to continue processing
        active = ~illegal_move

        # Player performs the action
        self.game_state[active, action[active]] = 1

        # Check if done (player 1 is always last in tied game)
        is_done = cp.all(self.game_state > 0, axis=1)

        # Update winners
        self.check_win()

        # Determine games that are done
        done = ((self._winners > 0) | is_done) & active

        # Map winners to rewards
        reward_mapping = cp.array([0, 1, -1])  # Index corresponds to winner number
        self.reward[done] = reward_mapping[self._winners[done]]
        self.terminated[done] = True
        self.truncated[done] = False

        # Update active games
        active = active & (~done)

        # Random move by the opponent for active games
        if cp.any(active):
            active_indices = indices[active]
            # Get available positions
            available_positions = self.game_state[active] == 0
            num_available = cp.sum(available_positions, axis=1)

            # Generate random indices for selection
            random_indices = cp.floor(
                cp.random.rand(active_indices.size) * num_available
            ).astype(cp.int32)

            # Compute opponent actions
            cumsum_positions = cp.cumsum(available_positions, axis=1)
            opponent_actions = cp.zeros(active_indices.size, dtype=cp.int32)
            for i in range(active_indices.size):
                pos = cp.searchsorted(cumsum_positions[i], random_indices[i] + 1)
                opponent_actions[i] = pos
            self.game_state[active_indices, opponent_actions] = 2

            # Update winners after opponent's move
            self.check_win()

            # Check if opponent won
            opponent_won = (self._winners > 0) & active
            self.reward[opponent_won] = -1
            self.terminated[opponent_won] = True
            self.truncated[opponent_won] = False

            # Update active games
            active = active & (~opponent_won)

            # For remaining active games
            self.reward[active] = 0
            self.terminated[active] = False
            self.truncated[active] = False

        # Calculate observations
        self.calc_obs()
        return self.observation, self.reward, self.terminated, self.truncated, self.info


def test():
    env = TicTacToeEnv()
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
    env = TicTacToeEnv(batch_size=dims)
    pre_time = time.time()
    num_steps = 0
    rng = cp.random.default_rng()
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
    for i in range(25, 500, 25):
        print(i)
        speed_test(i)

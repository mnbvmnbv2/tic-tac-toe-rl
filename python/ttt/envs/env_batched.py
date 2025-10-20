import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

import gymnasium


class GameStateSetting(Enum):
    FLAT = 0
    GRID = 1


class WinCheck(Enum):
    INDEXED = 0
    GRID = 1
    LOOP = 2


class StepMode(Enum):
    MASKS = 0
    LOOP = 1
    GRID = 2


@dataclass
class Settings:
    game_state: GameStateSetting = GameStateSetting.FLAT
    win_check: WinCheck = WinCheck.INDEXED
    step_mode: StepMode = StepMode.MASKS
    batch_size: int = 1


class TicTacToeEnv:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.uint8
        )
        self.action_space = gymnasium.spaces.Discrete(9)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1
        self.render_mode = None

        self._batch_size = self.settings.batch_size

        if self.settings.game_state == GameStateSetting.FLAT:
            self.game_state = np.zeros((self._batch_size, 9), dtype=np.uint8)
        elif self.settings.game_state == GameStateSetting.GRID:
            self.game_state = np.zeros((self._batch_size, 3, 3), dtype=np.uint8)
        else:
            raise ValueError("Invalid game state setting")

        self.reward = np.zeros(self._batch_size, dtype=np.float32)
        self.terminated = np.zeros(self._batch_size, dtype=bool)
        self.truncated = np.zeros(self._batch_size, dtype=bool)
        self.info = [{} for _ in range(self._batch_size)]
        self.observation = np.zeros((self._batch_size, 18), dtype=np.uint8)

        self._winners = np.zeros(self._batch_size, dtype=np.uint8)

    def get_obs(self) -> np.ndarray:
        if self.settings.game_state == GameStateSetting.FLAT:
            self.observation = (
                np.eye(3)[self.game_state][:, :, 1:]
                .flatten()
                .reshape(self._batch_size, 18)
            )
            return self.observation
        elif self.settings.game_state == GameStateSetting.GRID:
            self.observation = self.game_state.ravel()
            return self.observation
        else:
            raise ValueError("Invalid game state setting")

    def check_win(self) -> None:
        if self.settings.win_check == WinCheck.GRID:
            self.check_win_grid()
        elif self.settings.win_check == WinCheck.LOOP:
            self.check_win_loops()
        else:
            raise ValueError("Invalid game state setting")

    def check_win_grid(self) -> None:
        reshaped_state = self.game_state.reshape(self._batch_size, 3, 3)
        # 0 for tie, 1 for player 1, 2 for player 2
        # rows
        rows_equal = np.all(reshaped_state == reshaped_state[:, :, [0]], axis=2) & (
            reshaped_state[:, :, 0] != 0
        )
        row_winners = reshaped_state[:, :, 0] * rows_equal
        cols_equal = np.all(reshaped_state == reshaped_state[:, [0], :], axis=1) & (
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

        self._winners = np.maximum(
            row_winners.max(axis=1),
            np.maximum(
                col_winners.max(axis=1),
                np.maximum(diagonal_1_winners, diagonal_2_winners),
            ),
        )

    def check_win_loops(self) -> None:
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
        if self.settings.game_state == GameStateSetting.FLAT:
            self.game_state[game_idx] = np.zeros(9, dtype=np.uint8)
        elif self.settings.game_state == GameStateSetting.GRID:
            self.game_state[game_idx] = np.zeros((3, 3), dtype=np.uint8)
        else:
            raise ValueError("Invalid game state setting")
        self.reward[game_idx] = 0
        self.terminated[game_idx] = False
        self.truncated[game_idx] = False
        self.info[game_idx] = {}
        # could calculate observation here, but it would be unused

    def reset_all(self, seed=None) -> tuple[np.ndarray, dict]:
        # obs, info
        for g in range(self._batch_size):
            self._reset(g)

        return self.get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.settings.step_mode == StepMode.MASKS:
            return self.step_masks(action)
        elif self.settings.step_mode == StepMode.LOOP:
            return self.step_loop(action)
        elif self.settings.step_mode == StepMode.GRID:
            return self.step_3x3(action)
        else:
            raise ValueError("Invalid step mode")

    def step_masks(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        batch_size = self._batch_size
        indices = np.arange(batch_size)

        # Initialize arrays for rewards and statuses
        self.reward = np.zeros(batch_size, dtype=float)
        self.terminated = np.zeros(batch_size, dtype=bool)
        self.truncated = np.zeros(batch_size, dtype=bool)

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
        is_done = np.all(self.game_state > 0, axis=1)

        # Update winners
        self.check_win()

        # Determine games that are done
        done = ((self._winners > 0) | is_done) & active

        # Map winners to rewards
        reward_mapping = np.array([0, 1, -1])  # Index corresponds to winner number
        self.reward[done] = reward_mapping[self._winners[done]]
        self.terminated[done] = True
        self.truncated[done] = False

        # Update active games
        active = active & (~done)

        # Random move by the opponent for active games
        if np.any(active):
            active_indices = indices[active]
            # Get available positions
            available_positions = self.game_state[active] == 0
            num_available = np.sum(available_positions, axis=1)

            # Generate random indices for selection
            random_indices = np.floor(
                np.random.rand(active_indices.size) * num_available
            ).astype(np.int32)

            # Compute opponent actions
            cumsum_positions = np.cumsum(available_positions, axis=1)
            opponent_actions = np.zeros(active_indices.size, dtype=np.int32)
            for i in range(active_indices.size):
                pos = np.searchsorted(cumsum_positions[i], random_indices[i] + 1)
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

        return self.get_obs(), self.reward, self.terminated, self.truncated, self.info

    def step_loop(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
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
                # winner can only be player 1 else it's a tie
                self.reward[g] = self._winners[g]
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

        return self.get_obs(), self.reward, self.terminated, self.truncated, self.info

    def step_3x3(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        # turn int into (x, y)
        for g in range(self._batch_size):
            player_action = np.array([action[g] // 3, action[g] % 3])
            # if illegal move
            if self.game_state[g, player_action[0], player_action[1]] > 0:
                self.reward[g] = -1
                self.terminated[g] = False
                self.truncated[g] = False
                self.info[g] = {}
            # else, player performs the player_action
            self.game_state[g, player_action[0], player_action[1]] = 1

            # check if done (player 1 is always last in tied game)
            is_done = np.all(self.game_state[g, :] > 0)
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
                self.info[g] = {}

            # random move by the opponent
            opponent_action = np.where(self.game_state[g, :] == 0)[0]
            opponent_action = np.random.choice(opponent_action)
            self.game_state[g, opponent_action] = 2
            self.check_win()
            if self._winners[g] > 0:
                # only player 2 can win here
                self.reward[g] = -1
                self.terminated[g] = True
                self.truncated[g] = False
                self.info[g] = {}
            # else we continue the game
        return self.get_obs(), self.reward, self.terminated, self.truncated, {}


def speed_test(settings: Settings) -> None:
    print(f"Running speed test for {settings}")
    env = TicTacToeEnv(settings)
    num_steps = 0
    rng = np.random.default_rng()
    env.reset_all()
    pre_time = time.time()
    while time.time() - pre_time < 1:
        action = rng.integers(9, size=env._batch_size)
        obs, reward, terminated, truncated, info = env.step(action)
        num_steps += 1
        # restart the terminated or truncated games
        for g in range(settings.batch_size):
            if terminated[g] or truncated[g]:
                env._reset(g)

    print(f"Ran {num_steps * settings.batch_size} steps in 1 second")


if __name__ == "__main__":
    for i in [1, 100]:
        flat_grid_loop = Settings(
            game_state=GameStateSetting.FLAT,
            win_check=WinCheck.GRID,
            step_mode=StepMode.LOOP,
            batch_size=i,
        )
        speed_test(flat_grid_loop)
        flat_grid_masks = Settings(
            game_state=GameStateSetting.FLAT,
            win_check=WinCheck.GRID,
            step_mode=StepMode.MASKS,
            batch_size=i,
        )
        speed_test(flat_grid_masks)
        grid_grid_grid = Settings(
            game_state=GameStateSetting.GRID,
            win_check=WinCheck.GRID,
            step_mode=StepMode.GRID,
            batch_size=i,
        )
        speed_test(grid_grid_grid)

import time

import chex
from flax import struct
import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    board: jnp.ndarray


step_return = tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, dict]


@struct.dataclass
class EnvParams(environment.EnvParams):
    rew_win: int = 1
    rew_loss: int = -1
    rew_tie: int = 0
    rew_illegal: int = -1


class TicTacToeEnv(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        self.obs_shape = (18,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        # flatten one-hot encoding
        return jnp.concatenate(
            [state.board == 1, state.board == 2], axis=0, dtype=jnp.int32
        )

    def check_win(self, state: EnvState) -> int:
        # Win conditions defined as indices on the board
        win_conditions = jnp.array(
            [
                (0, 1, 2),
                (3, 4, 5),
                (6, 7, 8),
                (0, 3, 6),
                (1, 4, 8),
                (2, 5, 8),
                (0, 4, 8),
                (2, 4, 6),
            ],
            dtype=jnp.int32,
        )

        def check_line(winner, line):
            line_win = lax.cond(
                (state.board[line[0]] == state.board[line[1]])
                & (state.board[line[1]] == state.board[line[2]])
                & (state.board[line[0]] != 0),
                lambda: state.board[line[0]].astype(jnp.int32),  # Ensuring int32 output
                lambda: jnp.array(0, dtype=jnp.int32),  # Ensuring int32 output
            )
            return jnp.maximum(winner, line_win), None

        # Use `jnp.array(0)` as the initial carry value, which represents "no winner"
        winner, _ = lax.scan(check_line, jnp.array(0), win_conditions)
        return winner  # Returns 1 if player wins, 2 if opponent wins, 0 otherwise

    def handle_illegal_move(self, params: EnvParams, state: EnvState) -> step_return:
        return self.get_obs(state), state, jnp.array(params.rew_illegal), True, {}

    def step_env(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams,
    ) -> step_return:
        # Check for illegal move

        # Check for illegal moves
        illegal_move = state.board[action[0]] > 0
        # self.reward[illegal_move] = -1
        # self.terminated[illegal_move] = False
        # self.truncated[illegal_move] = False

        # Player performs the action
        player_moved_board = state.board.at(action[0]).set(1)

        # Check if done (player 1 is always last in tied game)
        is_done = jnp.all(player_moved_board > 0)

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

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> tuple[chex.Array, EnvState]:
        state = EnvState(time=0, board=jnp.zeros((9), dtype=jnp.int32))
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        winner = self.check_win(state)
        return jnp.logical_or(winner > 0, jnp.all(state.board > 0, axis=1))

    @property
    def name(self) -> str:
        return "TicTacToe"

    @property
    def num_actions(self) -> int:
        return 9

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(9)

    def observation_space(self, params: environment.EnvParams) -> spaces.Box:
        return spaces.Box(low=0, high=2, shape=(18,))

    def state_space(self, params: environment.EnvParams) -> spaces.Box:
        return spaces.Box(low=0, high=2, shape=(9,))


# Define function to perform step speed test
def step_speed_test(rng, env, state, env_params, duration=1.0):
    rng, reset_rng, step_rng, action_rng = jax.random.split(rng, 4)
    env.reset_env(reset_rng, env_params)
    # Warm-up to compile the function
    pre_compile = time.time()
    jit_step = jax.jit(env.step_env)
    action = jax.random.randint(action_rng, 1, 0, env.num_actions)
    _ = jit_step(step_rng, state, action, env_params)
    env.reset_env(reset_rng, env_params)
    print(f"Compilation time: {time.time() - pre_compile:.4f}s")

    start_time = time.time()
    steps = 0
    while time.time() - start_time < duration:
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        state = jit_step(step_rng, state, action, env_params)[1]
        steps += 1
    print(f"Ran {steps} steps in {duration:.4f}s")


if __name__ == "__main__":
    print(jax.devices())
    # test()
    env, env_params = TicTacToeEnv(), EnvParams()
    state = EnvState(time=0, board=jnp.zeros((9), dtype=jnp.int32))

    rng = jax.random.PRNGKey(0)
    step_speed_test(rng, env, state, env_params)

import time

import chex
from flax import struct
import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces

jax.config.update("jax_enable_x64", True)


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
        self._step = -1

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        # flatten one-hot encoding
        return jnp.concatenate(
            [state.board == 1, state.board == 2], axis=0, dtype=jnp.int32
        )

    def check_win(self, board: jnp.ndarray) -> int:
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
                (board[line[0]] == board[line[1]])
                & (board[line[1]] == board[line[2]])
                & (board[line[0]] != 0),
                lambda: board[line[0]].astype(jnp.int32),  # Ensuring int32 output
                lambda: jnp.array(0, dtype=jnp.int32),  # Ensuring int32 output
            )
            return jnp.maximum(winner, line_win), None

        # Use `jnp.array(0)` as the initial carry value, which represents "no winner"
        winner, _ = lax.scan(check_line, jnp.array(0), win_conditions)
        return winner  # Returns 1 if player wins, 2 if opponent wins, 0 otherwise

    def handle_illegal_move(self, params: EnvParams, state: EnvState) -> step_return:
        return self.get_obs(state), state, jnp.array(params.rew_illegal), True, {}

    def step_env(
        self, key: jax.random.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> step_return:
        self._step += 1
        # Check for illegal move
        illegal_move = state.board[action] != 0
        return lax.cond(
            illegal_move,
            lambda: self.handle_illegal_move(params, state),
            lambda: self.perform_player_move(key, state, action, params),
        )

    def perform_player_move(
        self, key: jax.random.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> step_return:
        # Update board with player move
        new_board = state.board.at[action].set(1)
        new_state = EnvState(self._step, new_board)
        winner = self.check_win(new_board)

        # Check if player won or if it's a tie
        player_wins = winner == 1
        is_tie = jnp.all(new_board > 0)

        return lax.cond(
            player_wins,
            lambda: (
                self.get_obs(new_state),
                new_state,
                jnp.array(params.rew_win),
                True,
                {},
            ),
            lambda: lax.cond(
                is_tie,
                lambda: (
                    self.get_obs(new_state),
                    new_state,
                    jnp.array(params.rew_tie),
                    True,
                    {},
                ),
                lambda: self.perform_opponent_move(key, new_state, params),
            ),
        )

    def perform_opponent_move(
        self, key: jax.random.PRNGKey, state: EnvState, params: EnvParams
    ) -> step_return:
        # Create a mask for legal moves (1 for legal, 0 for illegal)
        legal_moves_mask = (state.board == 0).astype(jnp.float32)

        # Count available moves
        num_legal_moves = legal_moves_mask.sum()

        # Ensure there's at least one legal move to prevent division errors
        def choose_move(_):
            # Randomly select a legal move
            probabilities = legal_moves_mask / num_legal_moves
            return jax.random.choice(key, a=jnp.arange(9), p=probabilities)

        opponent_action = lax.cond(
            num_legal_moves > 0,
            choose_move,
            lambda _: -1,  # Return -1 if no legal moves are available (should not happen in normal play)
            operand=None,
        )

        # Update board with opponent move if a valid move was chosen
        new_board = lax.cond(
            opponent_action >= 0,
            lambda action: state.board.at[action].set(2),
            lambda _: state.board,
            operand=opponent_action,
        )
        new_state = EnvState(self._step, new_board)
        winner = self.check_win(new_board)

        # Check if opponent won or if it's a tie
        opponent_wins = winner == 2
        is_tie = jnp.all(new_board > 0)

        return lax.cond(
            opponent_wins,
            lambda: (
                self.get_obs(new_state),
                new_state,
                jnp.array(params.rew_loss),
                True,
                {},
            ),
            lambda: lax.cond(
                is_tie,
                lambda: (
                    self.get_obs(new_state),
                    new_state,
                    jnp.array(params.rew_tie),
                    True,
                    {},
                ),
                lambda: (self.get_obs(new_state), new_state, jnp.array(0), False, {}),
            ),
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> tuple[chex.Array, EnvState]:
        self._step = -1
        state = EnvState(time=0, board=jnp.zeros((9), dtype=jnp.int32))
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        winner = check_win(state)
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


@jax.jit
def check_win(state) -> int:
    reshaped_state = state.board.reshape(3, 3)
    # 0 for tie, 1 for player 1, 2 for player 2
    # Check for row-wise winners
    rows_equal = jnp.all(reshaped_state == reshaped_state[:, [0]], axis=1) & (
        reshaped_state[:, 0] != 0
    )
    row_winners = reshaped_state[:, 0] * rows_equal

    # Check for column-wise winners
    cols_equal = jnp.all(reshaped_state == reshaped_state[[0], :], axis=0) & (
        reshaped_state[0, :] != 0
    )
    col_winners = reshaped_state[0, :] * cols_equal

    # Check for diagonal winners
    diagonal_1 = (
        (reshaped_state[0, 0] != 0)
        & (reshaped_state[0, 0] == reshaped_state[1, 1])
        & (reshaped_state[0, 0] == reshaped_state[2, 2])
    )
    diagonal_1_winners = reshaped_state[0, 0] * diagonal_1

    diagonal_2 = (
        (reshaped_state[0, 2] != 0)
        & (reshaped_state[0, 2] == reshaped_state[1, 1])
        & (reshaped_state[0, 2] == reshaped_state[2, 0])
    )
    diagonal_2_winners = reshaped_state[0, 2] * diagonal_2

    # Return the maximum winner (1 for player 1, 2 for player 2, 0 for no winner)
    return jnp.maximum(
        row_winners.max(),
        jnp.maximum(
            col_winners.max(),
            jnp.maximum(diagonal_1_winners, diagonal_2_winners),
        ),
    )


# Define function to perform step speed test
def step_speed_test(rng, step_fn, env, state, env_params, duration=1.0):
    # Warm-up to compile the function
    rng, step_rng, action_rng = jax.random.split(rng, 3)
    action = env.action_space(env_params).sample(action_rng)
    _ = step_fn(step_rng, state, action, env_params)

    start_time = time.time()
    steps = 0
    while time.time() - start_time < duration:
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        state = step_fn(step_rng, state, action, env_params)[1]
        steps += 1
    return steps


def test():
    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)

    # Instantiate the environment and its settings
    env, env_params = TicTacToeEnv(), EnvParams()

    # Reset the environment
    obs, state = env.reset(key_reset, env_params)

    # Perform speed test
    num_steps_regular = step_speed_test(rng, env.step, env, state, env_params)
    print(f"Number of regular steps in 1 second: {num_steps_regular}")


# Define function to perform a batched step speed test
def batched_step_speed_test(rng, step_fn, env, states, env_params, duration=1.0):
    # Number of parallel environments
    num_envs = states.board.shape[0]

    # Warm-up to compile the function with a batch of actions and keys
    rng, step_rng = jax.random.split(rng)
    action_rngs = jax.random.split(step_rng, num_envs)
    actions = jax.vmap(env.action_space(env_params).sample)(action_rngs)
    step_keys = jax.random.split(
        step_rng, num_envs
    )  # Batched keys for each environment
    _ = step_fn(step_keys, states, actions, env_params)

    start_time = time.time()
    steps = 0

    # Loop until the time duration is reached
    while time.time() - start_time < duration:
        rng, step_rng = jax.random.split(rng)
        action_rngs = jax.random.split(step_rng, num_envs)
        actions = jax.vmap(env.action_space(env_params).sample)(action_rngs)

        # Update all environments in parallel using vmap and batched PRNG keys
        step_keys = jax.random.split(
            step_rng, num_envs
        )  # Generate new keys for each step
        states = step_fn(step_keys, states, actions, env_params)[1]
        steps += 1

    # Multiply by batch size to get total steps across all environments
    return steps * num_envs


def batched_test(num_envs=12800000, duration=1.0):
    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)

    # Instantiate the environment and its parameters
    env, env_params = TicTacToeEnv(), EnvParams()

    # Initialize batched states for the specified number of environments
    obs, states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key_reset, num_envs), env_params
    )

    # Vectorize the environment step function for batched processing
    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Perform batched speed test
    num_steps_batched = batched_step_speed_test(
        rng, batched_step, env, states, env_params, duration
    )
    print(
        f"Number of batched steps across {num_envs} environments in {duration} second(s): {num_steps_batched}"
    )


if __name__ == "__main__":
    print(jax.devices())
    # test()
    batched_test()

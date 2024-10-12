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


@struct.dataclass
class EnvParams(environment.EnvParams):
    rew_win: int = 1
    rew_loss: int = -1
    rew_tie: int = 0
    rew_illegal: int = -1


class TicTacToeEnvSingle(environment.Environment[EnvState, EnvParams]):
    def __init__(self) -> None:
        super().__init__()
        self.obs_shape = (18,)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int | float | chex.Array,
        params: EnvParams,
    ) -> tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        # Check for illegal move
        illegal_move = state.board[action] > 0

        def handle_illegal_move(state):
            reward = jnp.array(params.rew_illegal)
            done = False
            return (
                lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                reward,
                done,
                {},
            )

        def handle_legal_move(state):
            # Player performs the action
            new_board = state.board.at[action].set(1)
            new_state = state.replace(board=new_board)

            # Check if the player has won or if it's a tie
            winner = check_win(new_state)
            is_tie = jnp.all(new_board > 0)
            game_over = (winner == 1) | is_tie

            def handle_player_win_or_tie(new_state):
                reward = jnp.where(
                    winner == 1,
                    params.rew_win,  # Player wins
                    params.rew_tie,  # Tie
                )
                return (
                    lax.stop_gradient(self.get_obs(new_state)),
                    lax.stop_gradient(new_state),
                    reward,
                    True,
                    {},
                )

            def handle_opponent_turn(new_state):
                # Generate all possible indices
                all_indices = jnp.arange(9, dtype=jnp.uint8)

                # Create a mask where legal moves are 1 and illegal moves are 0
                legal_moves = (new_state.board == 0).astype(jnp.float32)

                # Compute probabilities for legal moves
                total_legal_moves = legal_moves.sum()
                probabilities = legal_moves / total_legal_moves

                # Split the key for randomness
                key_opponent, key_next = jax.random.split(key)

                # Randomly select an index among legal moves
                random_index = jax.random.choice(
                    key_opponent, a=all_indices, p=probabilities
                )

                # Opponent performs the action
                new_board = new_state.board.at[random_index].set(2)
                new_state = new_state.replace(board=new_board)

                # Check if the opponent has won or if it's a tie
                winner_after_opponent = check_win(new_state)
                is_tie_after_opponent = jnp.all(new_board > 0)
                game_over = (winner_after_opponent == 2) | is_tie_after_opponent

                def handle_opponent_win_or_tie(new_state):
                    reward = jnp.where(
                        winner_after_opponent == 2,
                        params.rew_loss,  # Opponent wins
                        params.rew_tie,  # Tie
                    )
                    return (
                        lax.stop_gradient(self.get_obs(new_state)),
                        lax.stop_gradient(new_state),
                        reward,
                        True,
                        {},
                    )

                def continue_game(new_state):
                    reward = jnp.array(0)
                    done = False
                    return (
                        lax.stop_gradient(self.get_obs(new_state)),
                        lax.stop_gradient(new_state),
                        reward,
                        done,
                        {},
                    )

                return lax.cond(
                    game_over,
                    handle_opponent_win_or_tie,
                    continue_game,
                    new_state,
                )

            return lax.cond(
                game_over,
                handle_player_win_or_tie,
                handle_opponent_turn,
                new_state,
            )

        # Use `lax.cond` to handle illegal and legal moves
        return lax.cond(
            illegal_move,
            handle_illegal_move,
            handle_legal_move,
            state,
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> tuple[chex.Array, EnvState]:
        state = EnvState(time=0, board=jnp.zeros((9), dtype=jnp.uint8))
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        # flatten one-hot encoding
        return jnp.concatenate(
            [state.board == 1, state.board == 2], axis=0, dtype=jnp.uint8
        )

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
def step_speed_test(step_fn, env, state, action, env_params, key_step, duration=1.0):
    start_time = time.time()
    steps = 0
    while time.time() - start_time < duration:
        # Perform one step transition
        key_step, subkey = jax.random.split(key_step)
        state = step_fn(subkey, state, action, env_params)[
            1
        ]  # Only keep the updated state
        steps += 1
    return steps


# Perform vectorized (batched) speed test
def batched_speed_test(
    step_fn,
    state_batched,
    action,
    env_params,
    keys_batched,
    duration=1.0,
):
    start_time = time.time()
    steps = 0

    # Get the batch size from the state_batched.board
    batch_size = state_batched.board.shape[0]

    # Broadcast the action to match the batch size
    batched_action = jnp.broadcast_to(action, (batch_size,))

    # Define the key splitting function
    def split_fn(key):
        key1, key2 = jax.random.split(key)
        return key1, key2

    # Vectorize key splitting using jax.vmap
    split_keys = jax.vmap(split_fn)

    while time.time() - start_time < duration:
        # Split the keys in a batched way
        subkeys, keys_batched = split_keys(keys_batched)  # Split keys for the batch

        # Perform batched step transition
        state_batched = step_fn(subkeys, state_batched, batched_action, env_params)[
            1
        ]  # Only keep updated states
        steps += batch_size  # Increment by batch size
    return steps


def test():
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment and its settings
    env, env_params = TicTacToeEnvSingle(), EnvParams()

    # Reset the environment
    obs, state = env.reset(key_reset, env_params)

    # Sample a random action
    action = env.action_space(env_params).sample(key_act)

    # Perform speed test for non-JIT environment step
    num_steps_regular = step_speed_test(
        env.step, env, state, action, env_params, key_step
    )
    print(f"Number of regular steps in 1 second: {num_steps_regular}")

    # JIT-accelerated step transition
    jit_step = jax.jit(env.step)

    # Perform speed test for JIT-accelerated environment step
    num_steps_jit = step_speed_test(jit_step, env, state, action, env_params, key_step)
    print(f"Number of JIT steps in 1 second: {num_steps_jit}")

    # Batch environment reset and step using vmap
    batch_size = 10  # Number of environments in parallel
    batched_keys_reset = jax.random.split(key_reset, batch_size)
    batched_keys_step = jax.random.split(key_step, batch_size)

    # Vectorized environment reset and step using vmap
    reset_batched = jax.vmap(env.reset, in_axes=(0, None))
    step_batched = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Reset batched environments
    obs_batched, state_batched = reset_batched(batched_keys_reset, env_params)

    # Perform batched (vmap) speed test
    num_steps_batched = batched_speed_test(
        step_batched, state_batched, action, env_params, batched_keys_step
    )
    print(f"Number of batched (vmap) steps in 1 second: {num_steps_batched}")


if __name__ == "__main__":
    key = jax.random.key(1)
    print(jax.devices())
    # test()

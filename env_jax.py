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
        if illegal_move:
            reward = jnp.array(params.rew_illegal)
            done = False
            return (
                lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                reward,
                done,
                {},
            )

        # Player performs the action
        state.board[action] = 1

        # Check if done (player 1 is always last in tied game)
        is_done = jnp.all(state.board > 0, axis=1)
        # Update winner
        winner = check_win(state)

        # Determine games that are done
        done = winner > 0 or is_done

        if done:
            if winner == 0:
                reward = params.rew_tie
            else:
                reward = jnp.where(winner == 1, params.rew_win, params.rew_loss)
            done = True
            return (
                lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                jnp.array(reward),
                done,
                {},
            )

        # Compute opponent actions
        zero_indices = jnp.where(state.board == 0)[0]
        key = jax.random.PRNGKey(0)
        random_index = jax.random.choice(key, zero_indices)
        state.board[random_index] = 2

        # Update winner after opponent's move
        winner = check_win()

        # Check if opponent won
        opponent_won = winner > 0

        if opponent_won:
            reward = jnp.array(params.rew_loss)
            done = True
            return (
                lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                reward,
                done,
                {},
            )

        reward = jnp.array(0)
        done = False
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {},
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
    # rows
    rows_equal = jnp.all(reshaped_state == reshaped_state[:, [0]], axis=2) & (
        reshaped_state[:, 0] != 0
    )
    row_winners = reshaped_state[:, 0] * rows_equal
    cols_equal = jnp.all(reshaped_state == reshaped_state[[0], :], axis=1) & (
        reshaped_state[0, :] != 0
    )
    col_winners = reshaped_state[0, :] * cols_equal
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

    return jnp.maximum(
        row_winners.max(axis=1),
        jnp.maximum(
            col_winners.max(axis=1),
            jnp.maximum(diagonal_1_winners, diagonal_2_winners),
        ),
    )


def test():
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

    # Instantiate the environment & its settings.
    env, env_params = TicTacToeEnvSingle(), EnvParams()

    # Reset the environment.
    obs, state = env.reset(key_reset, env_params)

    # Sample a random action.
    action = env.action_space(env_params).sample(key_act)

    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)


if __name__ == "__main__":
    key = jax.random.key(1)
    test()

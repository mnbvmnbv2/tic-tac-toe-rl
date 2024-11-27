import time

import chex
from flax import struct
import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from env_jax_helper import RolloutWrapper
import numpy as np
import gymnax
import functools


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
        # Gather the board values at the win condition indices
        lines = board[win_conditions]  # Shape: (8, 3)

        # Check if all elements in a line are equal and not zero
        lines_equal = (lines == lines[:, [0]]) & (lines[:, 0:1] != 0)
        winners = lines[:, 0] * jnp.all(lines_equal, axis=1)

        # Return the maximum winner (1 or 2 if there's a winner, 0 otherwise)
        return jnp.max(winners)

    def handle_illegal_move(self, params: EnvParams, state: EnvState) -> step_return:
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(params.rew_illegal),
            True,
            {},
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env_(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> step_return:
        # Check for illegal move
        return lax.cond(
            state.board[action] != 0,
            lambda: self.handle_illegal_move(params, state),
            lambda: self.perform_player_move(key, state, action, params),
        )

    def perform_player_move(
        self, key: jax.random.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> step_return:
        # Update board with player move
        new_board = state.board.at[action].set(1)
        new_state = EnvState(state.time, new_board)
        winner = self.check_win(new_state.board)

        # Check if player won or if it's a tie
        player_wins = winner == 1
        is_tie = jnp.all(new_board > 0)

        return lax.cond(
            player_wins,
            lambda: (
                lax.stop_gradient(self.get_obs(new_state)),
                lax.stop_gradient(new_state),
                jnp.array(params.rew_win),
                True,
                {},
            ),
            lambda: lax.cond(
                is_tie,
                lambda: (
                    lax.stop_gradient(self.get_obs(new_state)),
                    lax.stop_gradient(new_state),
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
        new_state = EnvState(state.time, new_board)
        winner = self.check_win(new_state.board)

        # Check if opponent won or if it's a tie
        opponent_wins = winner == 2
        is_tie = jnp.all(new_board > 0)

        return lax.cond(
            opponent_wins,
            lambda: (
                lax.stop_gradient(self.get_obs(new_state)),
                lax.stop_gradient(new_state),
                jnp.array(params.rew_loss),
                True,
                {},
            ),
            lambda: lax.cond(
                is_tie,
                lambda: (
                    lax.stop_gradient(self.get_obs(new_state)),
                    lax.stop_gradient(new_state),
                    jnp.array(params.rew_tie),
                    True,
                    {},
                ),
                lambda: (self.get_obs(new_state), new_state, jnp.array(0), False, {}),
            ),
        )

    # _experimental
    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> step_return:
        # Check for illegal move
        illegal_move = state.board[action] > 0

        reward = jnp.where(illegal_move, params.rew_illegal, 0)

        # Player performs the action
        player_moved_board = state.board.at[action].set(1)

        # Check if done (player 1 is always last in tied game)
        is_done = jnp.all(player_moved_board > 0)

        # Update winners
        winner = self.check_win(player_moved_board)

        # Don't need to determine if game is done in gymnax configuration
        reward = jnp.where(
            jnp.logical_and(jnp.logical_not(illegal_move), winner == 1),
            params.rew_win,
            reward,
        )
        # Get available positions
        available_positions = player_moved_board == 0

        # Sample opponent action using Gumbel-max trick
        def sample_opponent_action(key, available_positions):
            u = jax.random.uniform(
                key, shape=available_positions.shape, minval=1e-6, maxval=1.0
            )
            gumbel_noise = -jnp.log(-jnp.log(u))
            masked_gumbel = jnp.where(available_positions, gumbel_noise, -jnp.inf)
            opponent_action = jnp.argmax(masked_gumbel)
            no_available_positions = jnp.all(jnp.logical_not(available_positions))
            opponent_action = jnp.where(no_available_positions, -1, opponent_action)
            return opponent_action

        opponent_action = sample_opponent_action(key, available_positions)

        # Apply opponent's move if valid
        opponent_moved_board = jax.lax.cond(
            opponent_action >= 0,
            lambda board: board.at[opponent_action].set(2),
            lambda board: board,
            player_moved_board,
        )

        # Update winners after opponent's move
        winner2 = self.check_win(opponent_moved_board)

        reward = jnp.where(
            jnp.logical_and(jnp.logical_not(is_done), winner2 == 2),
            params.rew_loss,
            reward,
        )

        state = EnvState(state.time + 1, opponent_moved_board)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            is_done,
            {},
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> tuple[chex.Array, EnvState]:
        state = EnvState(time=0, board=jnp.zeros((9), dtype=jnp.int32))
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        winner = self.check_win(state.board)
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


def speed_gymnax_random(env, env_params, num_envs, rng):
    manager = RolloutWrapper(
        env,
        env_params,
        # num_env_steps,
    )

    # Multiple rollouts for same network (different rng, e.g. eval)
    rng, rng_batch = jax.random.split(rng)
    rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
    if num_envs == 1:
        rollout_fn = manager.single_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(rng_batch_eval, None)
        steps_per_batch = obs.shape[0]
    else:
        rollout_fn = manager.batch_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(rng_batch_eval, None)
        steps_per_batch = obs.shape[0] * obs.shape[1]
    step_counter = 0

    start_t = time.time()
    # Loop over batch/single episode rollouts until steps are collected
    while time.time() < start_t + 1:
        rng, rng_batch = jax.random.split(rng)
        rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
        _ = rollout_fn(rng_batch_eval, None)
        step_counter += steps_per_batch
    return step_counter


if __name__ == "__main__":
    print(jax.devices())
    # test()
    # env, env_params = gymnax.make("CartPole-v1")
    env, env_params = TicTacToeEnv(), EnvParams()
    obs, state = env.reset_env(jax.random.PRNGKey(0), env_params)
    # env.step_env_experimental(jax.random.PRNGKey(0), state, 0, env_params)
    num_envs = 1
    # num_env_steps = 100_000

    rng = jax.random.PRNGKey(0)

    num_steps = []
    for run_id in range(5):
        rng, rng_run = jax.random.split(rng)
        num_steps_ = speed_gymnax_random(
            env,
            env_params,
            # num_env_steps,
            num_envs,
            rng_run,
        )

        # Store the computed SPSs
        print(f"Run {run_id + 1} - With {num_steps_} SPS")
        num_steps.append(num_steps_)
    print(num_steps)
    print(f"Mean SPS: {np.mean(num_steps)}")
    print(f"Std SPS: {np.std(num_steps)}")

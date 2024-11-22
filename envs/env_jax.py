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
        return lax.cond(
            state.board[action[0]] != 0,
            lambda: self.handle_illegal_move(params, state),
            lambda: self.perform_player_move(key, state, action, params),
        )

    def perform_player_move(
        self, key: jax.random.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> step_return:
        # Update board with player move
        new_board = state.board.at[action].set(1)
        new_state = EnvState(state.time, new_board)
        winner = self.check_win(new_state)

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
        new_state = EnvState(state.time, new_board)
        winner = self.check_win(new_state)

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

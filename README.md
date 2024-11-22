# tic-tac-toe-rl

## Goal

Components for fast RL.
- Fast base env
- Vectorization
- Fast training loop

It should also be easy to create, maintain and understand.

This repo is a test-project used to look at some parts of the RL stack to try to find components that
help to achieve the points above. 

## Env part

The test environment is TicTacToe which I thought would be easy, but the fact that it is a sequential MARL env
makes it a bit more challenging.

The env is simplified to a Single Agent env where the player is always player 1 and the opponent is a random policy.

- Numpy
- Numpy naive env batching
- Numba
- Cython
- Cupy
- Jax (Craftax)
- Mojo
- Julia
- Rust

## Training part

- CleanRL
- LeanRL
- PureJAXrl
- PufferEnv

## Testing/Benchmarking

- Ease of compatibility
- SPS with random policy
- Training speed
- ...

## Other

For UV cython compilation:

`uv run python setup.py build_ext --inplace`
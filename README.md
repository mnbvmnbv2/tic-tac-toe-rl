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

I have also made a "PVP" version where there are two agents and it truly is a sequenntial MA-env.

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
- Pytorch RL
- Stablebaselines 3 (meh)
- Ray Rllib (meh)
- Flux
- Burn
- Candle
- tch-rs

## Testing/Benchmarking

- Ease of compatibility
- SPS with random policy
- Training speed
- ...

## Other

For UV cython compilation:

`uv run python setup.py build_ext --inplace`
Then you need to move the generated files into the folders for correct import.

For testing
`uv run pytest`

For python bindings rust
`cd rust/python_bindings | uv run maturin develop`
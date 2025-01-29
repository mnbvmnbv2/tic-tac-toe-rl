from pvp.c_tictactoe_pvp_py import TicTacToeEnvPy, Settings

import numpy as np


def test_main():
    env = TicTacToeEnvPy(Settings(batch_size=3))
    state, info = env.reset_all()

    assert np.array_equal(
        state,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert info == [{}, {}, {}]

    # --- 1 ---
    state, reward, done, info = env.step(np.array([0, 2, 8], dtype=np.int16))

    assert np.array_equal(
        state,
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(
        reward,
        np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(done, np.array([0, 0, 0], dtype=np.int16))

    # --- 2 ---
    state, reward, done, info = env.step(np.array([1, 2, 1], dtype=np.int16))

    assert np.array_equal(
        state,
        np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(
        reward,
        np.array(
            [
                [0, 0],
                [0, -1],
                [0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(done, np.array([0, 0, 0], dtype=np.int16))

    # --- 3 ---
    state, reward, done, info = env.step(np.array([2, 8, 3], dtype=np.int16))

    assert np.array_equal(
        state,
        np.array(
            [
                [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(
        reward,
        np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(done, np.array([0, 0, 0], dtype=np.int16))

    # --- 4 ---
    state, reward, done, info = env.step(np.array([7, 8, 6], dtype=np.int16))

    assert np.array_equal(
        state,
        np.array(
            [
                [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(
        reward,
        np.array(
            [
                [0, 0],
                [-1, 0],
                [0, 0],
            ],
            dtype=np.int16,
        ),
    )
    assert np.array_equal(done, np.array([0, 0, 0], dtype=np.int16))

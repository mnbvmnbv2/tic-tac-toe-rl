from pvp.c_tictactoe_pvp_py import TicTacToeEnvPy, Settings

import numpy as np


def test_main():
    env = TicTacToeEnvPy(Settings(batch_size=1))
    state, info = env.reset_all()
    assert state.shape == (1, 19)
    assert info == [{}]

    state, reward, done, info = env.step(np.array([0], dtype=np.int16))

    assert state.shape == (1, 19)
    assert reward.shape == (1, 2)
    assert done.shape == (1,)
    assert info == [{}]

    assert np.array_equal(
        state,
        np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
            dtype=np.int16,
        ),
    )

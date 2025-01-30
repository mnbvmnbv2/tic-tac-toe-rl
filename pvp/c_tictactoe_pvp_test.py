from pvp.c_tictactoe_pvp_py import TicTacToeEnvPy, Settings

import numpy as np
import pytest

scenario_1 = {
    "name": "Simple 3-batch scenario",
    "batch_size": 3,
    "steps": [
        {
            "actions": np.array([0, 2, 8], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([1, 2, 1], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [0, -1],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([3, 8, 3], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([7, 8, 6], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [-1, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([6, 8, 4], dtype=np.int16),
            "expected_state": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [1, -1],
                    [-1, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([1, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([0, 2, 5], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, -1],
                    [-1, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([1, 0, 0], dtype=np.int16),
        },
    ],
}

all_scenarios = [scenario_1]


@pytest.mark.parametrize("scenario", all_scenarios, ids=lambda s: s["name"])
def test_tictactoe_scenario(scenario):
    env = TicTacToeEnvPy(Settings(batch_size=scenario["batch_size"]))

    state, info = env.reset_all()

    for step_idx, step_data in enumerate(scenario["steps"]):
        actions = step_data["actions"]
        exp_state = step_data["expected_state"]
        exp_reward = step_data["expected_reward"]
        exp_done = step_data["expected_done"]

        state, reward, done, info = env.step(actions)

        assert np.array_equal(state, exp_state), f"Mismatch in state at step {step_idx}"
        assert np.array_equal(
            reward, exp_reward
        ), f"Mismatch in reward at step {step_idx}"
        assert np.array_equal(done, exp_done), f"Mismatch in done at step {step_idx}"

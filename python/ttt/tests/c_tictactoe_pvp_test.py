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
        {
            "actions": np.array([3, 0, 2], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
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
            "actions": np.array([1, 8, 0], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
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
            "actions": np.array([4, 0, 7], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            "expected_done": np.array([0, 0, 1], dtype=np.int16),
        },
        {
            "actions": np.array([8, 2, 0], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
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
            "expected_done": np.array([0, 0, 1], dtype=np.int16),
        },
        {
            "actions": np.array([5, 3, 0], dtype=np.int16),
            "expected_state": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [-1, 1],
                    [0, 0],
                    [0, -1],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([1, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([5, 3, 1], dtype=np.int16),
            "expected_state": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 1],
                    [-1, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([1, 0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([0, 0, 0], dtype=np.int16),
            "expected_state": np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [-1, 0],
                    [-1, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0, 0], dtype=np.int16),
        },
    ],
}

scenario_2 = {
    "name": "Scenario 2 - partial moves with batch_size=2",
    "batch_size": 2,
    "steps": [
        {
            "actions": np.array([0, 2], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [0, 0],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0], dtype=np.int16),
        },
        {
            "actions": np.array([1, 2], dtype=np.int16),
            "expected_state": np.array(
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array(
                [
                    [0, 0],
                    [0, -1],
                ],
                dtype=np.int16,
            ),
            "expected_done": np.array([0, 0], dtype=np.int16),
        },
    ],
}

scenario_3 = {
    "name": "Scenario 3 - simple 1-env, 3 moves",
    "batch_size": 1,
    "steps": [
        {
            "actions": np.array([0], dtype=np.int16),
            "expected_state": np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        {
            "actions": np.array([4], dtype=np.int16),
            "expected_state": np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        {
            "actions": np.array([8], dtype=np.int16),
            "expected_state": np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
    ],
}

scenario_4 = {
    "name": "Scenario 4 - p1 wins on main diagonal",
    "batch_size": 1,
    "steps": [
        # Step 1 (Player 1 places at cell 0)
        {
            "actions": np.array([0], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 (p1=1, p2=0)
                        1,
                        0,
                        # square1..8 empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 1 (player2)
                        1,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 2 (Player 2 places at cell 1)
        {
            "actions": np.array([1], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => p2
                        0,
                        1,
                        # square2..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 0 (player1)
                        0,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 3 (Player 1 places at cell 4)
        {
            "actions": np.array([4], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => p2
                        0,
                        1,
                        # square2 => empty
                        0,
                        0,
                        # square3 => empty
                        0,
                        0,
                        # square4 => p1
                        1,
                        0,
                        # square5..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 1 (player2)
                        1,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 4 (Player 2 places at cell 2)
        {
            "actions": np.array([2], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => p2
                        0,
                        1,
                        # square2 => p2
                        0,
                        1,
                        # square3 => empty
                        0,
                        0,
                        # square4 => p1
                        1,
                        0,
                        # square5..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 0 (player1)
                        0,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 5 (Player 1 places at cell 8 -> wins main diagonal)
        {
            "actions": np.array([8], dtype=np.int16),
            "expected_state": np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[1, -1]], dtype=np.int16),
            "expected_done": np.array([1], dtype=np.int16),
        },
    ],
}

scenario_5 = {
    "name": "Scenario 5 - p2 wins on anti-diagonal (2,4,6)",
    "batch_size": 1,
    "steps": [
        # Step 1 (Player 1 at cell 0)
        {
            "actions": np.array([0], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 1 (player2)
                        1,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 2 (Player 2 at cell 2)
        {
            "actions": np.array([2], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => empty
                        0,
                        0,
                        # square2 => p2
                        0,
                        1,
                        # square3..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 0 (player1)
                        0,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 3 (Player 1 at cell 1)
        {
            "actions": np.array([1], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => p1
                        1,
                        0,
                        # square2 => p2
                        0,
                        1,
                        # square3..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 1
                        1,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 4 (Player 2 at cell 4)
        {
            "actions": np.array([4], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => p1
                        1,
                        0,
                        # square2 => p2
                        0,
                        1,
                        # square3 => empty
                        0,
                        0,
                        # square4 => p2
                        0,
                        1,
                        # square5..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 0
                        0,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 5 (Player 1 at cell 3)
        {
            "actions": np.array([3], dtype=np.int16),
            "expected_state": np.array(
                [
                    [
                        # square0 => p1
                        1,
                        0,
                        # square1 => p1
                        1,
                        0,
                        # square2 => p2
                        0,
                        1,
                        # square3 => p1
                        1,
                        0,
                        # square4 => p2
                        0,
                        1,
                        # square5..8 => empty
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        # next perspective = 1
                        1,
                    ]
                ],
                dtype=np.int16,
            ),
            "expected_reward": np.array([[0, 0]], dtype=np.int16),
            "expected_done": np.array([0], dtype=np.int16),
        },
        # Step 6 (Player 2 at cell 6 -> completes diagonal 2,4,6)
        {
            "actions": np.array([6], dtype=np.int16),
            "expected_state": np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.int16,
            ),
            # Because the final perspective is now 0 (player1),
            # the reward array is from player1's viewpoint: [-1, 1].
            # (p1 loses, p2 wins.)
            "expected_reward": np.array([[-1, 1]], dtype=np.int16),
            "expected_done": np.array([1], dtype=np.int16),
        },
    ],
}

all_scenarios = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]


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


def test_tictactoe_rollout():
    batch_size = 3
    num_steps = 13
    env = TicTacToeEnvPy(Settings(batch_size=batch_size))
    state, _ = env.reset_all()

    # Allocate rollout buffers for X and O.
    x_states = np.zeros((num_steps, batch_size, 19), dtype=np.float16)
    x_actions = np.zeros((num_steps, batch_size), dtype=np.int32)
    x_rewards = np.zeros((num_steps, batch_size), dtype=np.float32)
    x_dones = np.zeros((num_steps, batch_size), dtype=np.int16)
    # For each environment we will record transitions only on timesteps where it is X’s turn.
    x_ptr = np.zeros(batch_size, dtype=np.int32)

    o_states = np.zeros((num_steps, batch_size, 19), dtype=np.float16)
    o_actions = np.zeros((num_steps, batch_size), dtype=np.int32)
    o_rewards = np.zeros((num_steps, batch_size), dtype=np.float32)
    o_dones = np.zeros((num_steps, batch_size), dtype=np.int16)
    o_ptr = np.zeros(batch_size, dtype=np.int32)

    rollout_actions = [
        [0, 2, 8],
        [1, 2, 1],
        [3, 8, 3],
        [7, 8, 6],
        [6, 8, 4],
        [0, 2, 5],
        [3, 0, 2],
        [1, 8, 0],
        [4, 0, 7],
        [8, 2, 0],
        [5, 3, 0],
        [5, 3, 1],
        [0, 0, 0],
    ]

    reward = np.zeros((batch_size, 2), dtype=np.float32)
    done = np.zeros(batch_size, dtype=np.int16)

    for step_action in rollout_actions:
        player_turns = state[:, 18].astype(int)
        x_envs = np.where(player_turns == 0)[0]
        o_envs = np.where(player_turns == 1)[0]
        actions = np.array(step_action, dtype=np.int16)
        for e in x_envs:
            t = x_ptr[e]
            if t < num_steps:
                x_states[t, e] = state[e]
                x_actions[t, e] = actions[e]
                x_rewards[t, e] = reward[e, 0]
                x_dones[t, e] = done[e]
                x_ptr[e] += 1
        for e in o_envs:
            t = o_ptr[e]
            if t < num_steps:
                o_states[t, e] = state[e]
                o_actions[t, e] = actions[e]
                o_rewards[t, e] = reward[e, 1]
                o_dones[t, e] = done[e]
                o_ptr[e] += 1

        state, reward, done, _ = env.step(np.array(actions, dtype=np.int16))

    # Check buffer
    assert np.array_equal(
        x_states[:7, :],
        np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ],
                [
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                ],
                [
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ),
    )

    assert np.array_equal(
        o_states[:6, :],
        np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                ],
                [
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                ],
                [
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                ],
                [
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
            ],
            dtype=np.float16,
        ),
    )

import time
import numpy as np
from c_tictactoe import TicTacToeEnvSingle


def speed_test():
    game_states = np.zeros(18, dtype=np.int32)
    rewards = np.zeros(1, dtype=np.int32)
    done = np.zeros(1, dtype=np.int32)
    env = TicTacToeEnvSingle(
        game_states=game_states,
        rewards=rewards,
        done=done,
    )
    pre_time = time.time()
    num_steps = 0
    while time.time() - pre_time < 5:
        env.reset()
        while not done[0]:
            # print(done)
            # print(game_states)
            env.step(np.random.randint(9))
            num_steps += 1
    print(f"Ran {num_steps} steps in 5 seconds")


def other():
    pre_time = time.time()
    num_steps = 0
    while time.time() - pre_time < 5:
        num_steps += 1
    print(f"Ran {num_steps} steps in 5 seconds")


if __name__ == "__main__":
    other()

import time
import numpy as np
from envs.c_tictactoe import TicTacToeEnv


def speed_test(dim=1):
    game_states = np.zeros((dim, 18), dtype=np.int16)
    rewards = np.zeros((dim), dtype=np.int16)
    done = np.zeros((dim), dtype=np.int16)
    winners = np.zeros((dim), dtype=np.int16)
    env = TicTacToeEnv(
        game_states=game_states,
        rewards=rewards,
        done=done,
        winners=winners,
    )
    pre_time = time.time()
    num_steps = 0
    while time.time() - pre_time < 1:
        env.reset()
        while not done[0]:
            # print(done)
            # print(game_states)
            env.step(np.random.randint(9))
            num_steps += 1
    print(f"Ran {num_steps * dim} steps in 1 seconds")


def other():
    pre_time = time.time()
    num_steps = 0
    while time.time() - pre_time < 1:
        num_steps += 1
    print(f"Ran {num_steps} steps in 1 seconds")


if __name__ == "__main__":
    for i in range(1, 25):
        print(f"dim={i}")
        speed_test(i)

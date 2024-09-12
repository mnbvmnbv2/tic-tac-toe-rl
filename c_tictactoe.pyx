# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
from libc.stdlib cimport rand

cdef class TicTacToeEnvSingle:
    cdef:
        int[:] game_states
        int[:] rewards
        int[:] done
        #int[:] observations

    def __init__(
            self, 
            cnp.ndarray game_states, 
            cnp.ndarray rewards,
            cnp.ndarray done
            #cnp.ndarray observations, 
        ):
        self.game_states = game_states
        self.rewards = rewards
        self.done = done
        #self.observations = observations

    #cdef cnp.ndarray[cnp.uint8_t, ndim=1] compute_observations(self):
        # Flatten one-hot encoding
        #self.observations = self.game_state

    cdef int check_win(self):
        # 0 for tie, 1 for player 1, 2 for player 2
        cdef int i
        cdef int j
        # rows
        for j in range(2):
            for i in range(3):
                if (
                    self.game_states[i * 6 + j]
                    == self.game_states[i * 6 + 2 + j]
                    == self.game_states[i * 6 + 4 + j]
                    != 0
                ):
                    return j + 1
        # columns
        for j in range(2):
            for i in range(3):
                if (
                    self.game_states[2 * i + j]
                    == self.game_states[2 * i + 6 + j]
                    == self.game_states[2 * i + 12 + j]
                    != 0
                ):
                    return j + 1
        # diagonals
        for j in range(2):
            if self.game_states[0 + j] == self.game_states[6 + j] == self.game_states[12 + j] != 0:
                return j + 1
            if self.game_states[4 + j] == self.game_states[6 + j] == self.game_states[8 + j] != 0:
                return j + 1
            if self.game_states[0 + j] == self.game_states[4 + j] == self.game_states[8 + j] != 0:
                return j + 1
            if self.game_states[2 + j] == self.game_states[4 + j] == self.game_states[6 + j] != 0:
                return j + 1
        return 0

    cpdef cnp.ndarray reset(self):
        # obs, info
        self.game_states[:] = 0
        self.rewards[:] = 0
        self.done[0] = 0

    cpdef tuple step(self, int action):
        # obs, reward, terminated, truncated, info
        cdef int winner
        cdef int opponent_action
        # if illegal move
        if self.game_states[action * 2] > 0 or self.game_states[action * 2 + 1] > 0:
            self.rewards[0] = -1
        
        # else, player performs the action
        self.game_states[action * 2] = 1

        # check if done (player 1 is always last in tied game)
        self.done[0] = int(sum(self.game_states) == 9)
        winner = self.check_win()
        if winner > 0 or self.done[0]:
            if winner == 0:
                self.rewards[0] = 0
            elif winner == 1:
                self.rewards[0] = 1
            else:
                self.rewards[0] = -1
            self.done[0] = 1

        # Collect all available moves
        available_moves = []
        for i in range(9):
            if self.game_states[i * 2] == 0 and self.game_states[i * 2 + 1] == 0:
                available_moves.append(i)

        # If there are available moves, select a random move for the opponent
        if available_moves:
            opponent_action = available_moves[rand() % len(available_moves)]
            self.game_states[opponent_action * 2 + 1] = 1
            winner = self.check_win()
            if winner > 0:
                # only player 2 can win here
                self.rewards[0] = -1
                self.done[0] = 1

        # else we continue the game

    def nice_print(self):
        cdef int i
        rows = []
        for i in range(3):
            rows.append(" ".join([[" ", "X", "O"][x] for x in self.game_states[i * 3 : i * 3 + 3]]))
        return "\n".join(rows)
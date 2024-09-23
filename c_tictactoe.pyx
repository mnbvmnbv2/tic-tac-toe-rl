# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
from libc.stdlib cimport rand

cdef class TicTacToeEnvSingle:
    cdef:
        short[:, :] game_states
        short[:] rewards
        short[:] done
        short[:] winners
        #short[:, :] observations

    def __init__(
            self, 
            cnp.ndarray game_states, 
            cnp.ndarray rewards,
            cnp.ndarray done,
            cnp.ndarray winners,
            #cnp.ndarray observations,
        ):
        self.game_states = game_states
        self.rewards = rewards
        self.done = done
        self.winners = winners
        #self.observations = observations

    cdef void check_win(self):
        # 0 for tie, 1 for player 1, 2 for player 2
        cdef short i
        cdef short j
        cdef short batch_dim

        for batch_dim in range(self.game_states.shape[0]):
            # rows
            for j in range(2):
                for i in range(3):
                    if (
                        self.game_states[batch_dim, i * 6 + j]
                        == self.game_states[batch_dim, i * 6 + 2 + j]
                        == self.game_states[batch_dim, i * 6 + 4 + j]
                        != 0
                    ):
                        self.winners[batch_dim] = j + 1
            # columns
            for j in range(2):
                for i in range(3):
                    if (
                        self.game_states[batch_dim, 2 * i + j]
                        == self.game_states[batch_dim, 2 * i + 6 + j]
                        == self.game_states[batch_dim, 2 * i + 12 + j]
                        != 0
                    ):
                        self.winners[batch_dim] = j + 1
            # diagonals
            for j in range(2):
                if self.game_states[batch_dim, 0 + j] == self.game_states[batch_dim, 6 + j] == self.game_states[batch_dim, 12 + j] != 0:
                    self.winners[batch_dim] = j + 1
                if self.game_states[batch_dim, 4 + j] == self.game_states[batch_dim, 6 + j] == self.game_states[batch_dim, 8 + j] != 0:
                    self.winners[batch_dim] = j + 1
                if self.game_states[batch_dim, 0 + j] == self.game_states[batch_dim, 4 + j] == self.game_states[batch_dim, 8 + j] != 0:
                    self.winners[batch_dim] = j + 1
                if self.game_states[batch_dim, 2 + j] == self.game_states[batch_dim, 4 + j] == self.game_states[batch_dim, 6 + j] != 0:
                    self.winners[batch_dim] = j + 1
            if self.winners[batch_dim] == 0:
                self.winners[batch_dim] = 0 

    cpdef void reset(self):
        # obs, info
        self.game_states[:, :] = 0
        self.rewards[:] = 0
        self.done[:] = 0
        self.winners[:] = 0

    cpdef void step(self, short action):
        # obs, reward, terminated, truncated, info
        cdef short opponent_action
        cdef short batch_dim

        for batch_dim in range(self.game_states.shape[0]):
            # if illegal move
            if self.game_states[batch_dim, action * 2] > 0 or self.game_states[batch_dim, action * 2 + 1] > 0:
                self.rewards[batch_dim] = -1
            
            # else, player performs the action
            self.game_states[batch_dim, action * 2] = 1

            # check if done (player 1 is always last in tied game)
            self.done[batch_dim] = int(sum(self.game_states[batch_dim]) == 9)
            self.check_win()
            if self.winners[batch_dim] > 0 or self.done[batch_dim]:
                if self.winners[batch_dim] == 0:
                    self.rewards[batch_dim] = 0
                elif self.winners[batch_dim] == 1:
                    self.rewards[batch_dim] = 1
                else:
                    self.rewards[batch_dim] = -1
                self.done[batch_dim] = 1

            # Collect all available moves
            available_moves = []
            for i in range(9):
                if self.game_states[batch_dim, i * 2] == 0 and self.game_states[batch_dim, i * 2 + 1] == 0:
                    available_moves.append(i)

            # If there are available moves, select a random move for the opponent
            if available_moves:
                opponent_action = available_moves[rand() % len(available_moves)]
                self.game_states[batch_dim, opponent_action * 2 + 1] = 1
                self.check_win()
                if self.winners[batch_dim] > 0:
                    # only player 2 can win here
                    self.rewards[batch_dim] = -1
                    self.done[batch_dim] = 1

            # else we continue the game

    # def nice_print(self):
    #     cdef int i
    #     rows = []
    #     for i in range(3):
    #         rows.append(" ".join([[" ", "X", "O"][x] for x in self.game_states[i * 3 : i * 3 + 3]]))
    #     return "\n".join(rows)
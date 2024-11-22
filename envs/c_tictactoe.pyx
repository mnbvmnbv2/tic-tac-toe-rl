# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args=/openmp
# distutils: extra_link_args=/openmp
cimport cython
cimport numpy as cnp
from libc.stdlib cimport rand

cdef class TicTacToeEnv:
    cdef:
        short[:, ::1] game_states
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

    # @cython.boundscheck(False)  # Deactivate bounds checking
    # @cython.wraparound(False)   # Deactivate negative indexing.
    cdef void check_win(self):
        # 0 for tie, 1 for player 1, 2 for player 2
        cdef short i
        cdef short j
        cdef Py_ssize_t x
        cdef Py_ssize_t y
        cdef Py_ssize_t z
        cdef Py_ssize_t batch_dim

        for batch_dim in range(self.game_states.shape[0]):
            # rows
            for i in range(2):
                for j in range(3):
                    x = j * 6 + i
                    y = j * 6 + 2 + i
                    z = j * 6 + 4 + i
                    if (
                        self.game_states[batch_dim, x]
                        == self.game_states[batch_dim, y]
                        == self.game_states[batch_dim, z]
                        != 0
                    ):
                        self.winners[batch_dim] = i + 1
                        continue
                # columns
                for j in range(3):
                    x = j * 2 + i
                    y = j * 2 + 6 + i
                    z = j * 2 + 12 + i
                    if (
                        self.game_states[batch_dim, x]
                        == self.game_states[batch_dim, y]
                        == self.game_states[batch_dim, z]
                        != 0
                    ):
                        self.winners[batch_dim] = i + 1
                        continue
                # diagonals
                x = i
                y = i + 8
                z = i + 16
                if (
                    self.game_states[batch_dim, x]
                    == self.game_states[batch_dim, y]
                    == self.game_states[batch_dim, z]
                    != 0
                ):
                    self.winners[batch_dim] = i + 1
                    continue
                x = i + 4
                y = i + 8
                z = i + 12
                if (
                    self.game_states[batch_dim, x]
                    == self.game_states[batch_dim, y]
                    == self.game_states[batch_dim, z]
                    != 0
                ):
                    self.winners[batch_dim] = i + 1

    cpdef void reset_all(self):
        self.game_states[:, :] = 0
        self.rewards[:] = 0
        self.done[:] = 0
        self.winners[:] = 0

    cpdef void reset(self, Py_ssize_t idx):
        # obs, info
        self.game_states[idx, :] = 0
        self.rewards[idx] = 0
        self.done[idx] = 0
        self.winners[idx] = 0

    # @cython.boundscheck(False)  # Deactivate bounds checking
    # @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef void step(self, short[:] action):
        # obs, reward, terminated, truncated, info
        cdef short opponent_action
        cdef Py_ssize_t batch_dim
        cdef int[9] available_moves
        cdef int num_available_moves
        cdef int i
        cdef int num_moves_made
        cdef short current_action

        for batch_dim in range(self.game_states.shape[0]):
            current_action = action[batch_dim]
            # if illegal move
            if self.game_states[batch_dim, current_action * 2] > 0 or self.game_states[batch_dim, current_action * 2 + 1] > 0:
                self.rewards[batch_dim] = -1
                continue
            
            # else, player performs the current_action
            self.game_states[batch_dim, current_action * 2] = 1

            # check if done (player 1 is always last in tied game)
            for i in range(9):
                if self.game_states[batch_dim, i * 2] > 0 or self.game_states[batch_dim, i * 2 + 1] > 0:
                    num_moves_made += 1
            if num_moves_made == 9:
                self.done[batch_dim] = 1
            self.check_win()
            if self.winners[batch_dim] > 0 or self.done[batch_dim]:
                if self.winners[batch_dim] == 0:
                    self.rewards[batch_dim] = 0
                elif self.winners[batch_dim] == 1:
                    self.rewards[batch_dim] = 1
                else:
                    self.rewards[batch_dim] = -1
                self.done[batch_dim] = 1
                continue

            # Collect all available moves
            num_available_moves = 0
            for i in range(9):
                if self.game_states[batch_dim, i * 2] == 0 and self.game_states[batch_dim, i * 2 + 1] == 0:
                    available_moves[num_available_moves] = i
                    num_available_moves += 1

            # If there are available moves, select a random move for the opponent
            if num_available_moves > 0:
                opponent_action = available_moves[rand() % num_available_moves]
                self.game_states[batch_dim, opponent_action * 2 + 1] = 1
                self.check_win()
                if self.winners[batch_dim] > 0:
                    # only player 2 can win here
                    self.rewards[batch_dim] = -1
                    self.done[batch_dim] = 1

            # else we continue the game

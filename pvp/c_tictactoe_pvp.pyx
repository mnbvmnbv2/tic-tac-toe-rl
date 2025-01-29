# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args=/openmp
# distutils: extra_link_args=/openmp

cimport cython
cimport numpy as cnp

# State is length=19: 9 * 2 bits for board occupancy + 1 for current player (0 or 1).
# states[:, 0..17] => pairs for each cell:
#    cell i => (states[env, 2*i], states[env, 2*i+1])
# states[:, 18] => current player (0 if it's X's turn, 1 if it's O's turn).

cdef class TicTacToePVPEnv:
    cdef:
        short[:, ::1] states
        short[:, ::1] rewards
        short[:] done
        short[:] winners

    def __init__(
            self, 
            cnp.ndarray states, 
            cnp.ndarray rewards,
            cnp.ndarray done,
            cnp.ndarray winners,
        ):
        self.states = states
        self.rewards = rewards
        self.done = done
        self.winners = winners

    cdef void check_win(self, Py_ssize_t game):
        """
        0 for tie, 1 for player X, 2 for player O
        We interpret states[game, 2*i] as X in cell i, 
        and states[game, 2*i+1] as O in cell i.
        """
        cdef short i
        cdef short j
        cdef Py_ssize_t x
        cdef Py_ssize_t y
        cdef Py_ssize_t z

        # rows
        for i in range(2):
            for j in range(3):
                x = j * 6 + i
                y = j * 6 + 2 + i
                z = j * 6 + 4 + i
                if (
                    self.states[game, x]
                    == self.states[game, y]
                    == self.states[game, z]
                    != 0
                ):
                    self.winners[game] = i + 1
                    continue
            # columns
            for j in range(3):
                x = j * 2 + i
                y = j * 2 + 6 + i
                z = j * 2 + 12 + i
                if (
                    self.states[game, x]
                    == self.states[game, y]
                    == self.states[game, z]
                    != 0
                ):
                    self.winners[game] = i + 1
                    continue
            # diagonals
            x = i
            y = i + 8
            z = i + 16
            if (
                self.states[game, x]
                == self.states[game, y]
                == self.states[game, z]
                != 0
            ):
                self.winners[game] = i + 1
                continue
            x = i + 4
            y = i + 8
            z = i + 12
            if (
                self.states[game, x]
                == self.states[game, y]
                == self.states[game, z]
                != 0
            ):
                self.winners[game] = i + 1

    cpdef void reset_all(self):
        self.states[:, :] = 0
        self.rewards[:, :] = 0
        self.done[:] = 0
        self.winners[:] = 0

    cpdef void reset(self, Py_ssize_t idx):
        # obs, info
        self.states[idx, :] = 0
        self.rewards[idx, :] = 0
        self.done[idx] = 0
        self.winners[idx] = 0

    # @cython.boundscheck(False)  # Deactivate bounds checking
    # @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef void step(self, short[:] action):
        """
        action is shape (batch_size,) - one action per environment.
        We apply the move for the current player, ignoring illegal moves by giving -1 reward 
        but continuing until someone wins or the board is full.
        """
        cdef Py_ssize_t batch_dim
        cdef int i
        cdef int num_moves_made
        cdef short current_action

        for batch_dim in range(self.states.shape[0]):
            num_moves_made = 0
            if self.done[batch_dim]:
                # clear all before start of enxt game
                self.reset(batch_dim)

            current_action = action[batch_dim]
            # get players turn
            player = self.states[batch_dim, 18] # 0 for X, 1 for O
            # if illegal move
            if self.states[batch_dim, current_action * 2] > 0 or self.states[batch_dim, current_action * 2 + 1] > 0:
                # done
                self.rewards[batch_dim, player] = -1
                continue
            
            # else, player performs the current_action
            self.states[batch_dim, current_action * 2 + player] = 1
            # reset reward in case it previously was -1 for example
            self.rewards[batch_dim, player] = 0

            # switch player
            self.states[batch_dim, 18] = 1 - player

            # check if done (player 1 is always last in tied game)
            for i in range(9):
                if self.states[batch_dim, i * 2] > 0 or self.states[batch_dim, i * 2 + 1] > 0:
                    num_moves_made += 1
            if num_moves_made == 9:
                # done, board is filled
                self.done[batch_dim] = 1
                self.states[batch_dim, :] = 0

            self.check_win(batch_dim)
            if self.winners[batch_dim] > 0 or self.done[batch_dim]:
                if self.winners[batch_dim] == 0:
                    self.rewards[batch_dim, :] = 0
                elif self.winners[batch_dim] == 1:
                    # X won => reward X=+1, O=-1
                    self.rewards[batch_dim, 0] = 1
                    self.rewards[batch_dim, 1] = -1
                else:
                    # O won => O=+1, X=-1
                    self.rewards[batch_dim, 0] = -1
                    self.rewards[batch_dim, 1] = 1
                # done
                self.done[batch_dim] = 1
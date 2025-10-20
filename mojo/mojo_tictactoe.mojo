import random

from tensor import Tensor

struct TicTacToe:
    var board: Tensor[DType.int8] 

    fn __init__(inout self):
        self.board = Tensor[DType.int8](9)
        for i in range(9):
            self.board.__setitem__(i, 0)

    fn reset(inout self) -> Tensor[DType.int8]:
        for i in range(9):
            self.board.__setitem__(i, 0)
        return self.get_obs()

    fn get_obs(inout self) -> Tensor[DType.int8]:
        var obs: Tensor[DType.int8]  = Tensor[DType.int8](18)
        for i in range(9):
            if self.board.__getitem__(i) == 0:
                pass
            elif self.board.__getitem__(i) == 1:
                self.board.__setitem__(i * 2, 1)  # Player 1
            else:
                self.board.__setitem__(i * 2 + 1, 1)  # Player 2
        return obs

def main():
    var game = TicTacToe()
    obs = game.reset()
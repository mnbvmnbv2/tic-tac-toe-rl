extern crate rand;

use rand::Rng;

pub struct TicTacToeEnv {
    pub game_states: Vec<[i8; 18]>,
    pub rewards: Vec<i8>,
    pub done: Vec<bool>,
    pub winners: Vec<i8>,
}

impl TicTacToeEnv {
    pub fn new(n_envs: usize) -> Self {
        TicTacToeEnv {
            game_states: vec![[0; 18]; n_envs],
            rewards: vec![0; n_envs],
            done: vec![false; n_envs],
            winners: vec![0; n_envs],
        }
    }

    pub fn reset(&mut self) {
        for state in self.game_states.iter_mut() {
            *state = [0; 18];
        }
        for r in self.rewards.iter_mut() {
            *r = 0;
        }
        for d in self.done.iter_mut() {
            *d = false;
        }
        for w in self.winners.iter_mut() {
            *w = 0;
        }
    }

    pub fn reset_game(&mut self, idx: usize) {
        self.game_states[idx] = [0; 18];
        self.rewards[idx] = 0;
        self.done[idx] = false;
        self.winners[idx] = 0;
    }

    pub fn check_win(&mut self, game_idx: usize) {
        let state = &self.game_states[game_idx];
        for player in 0..2 {
            for line in [
                (0, 2, 4),
                (6, 8, 10),
                (12, 14, 16),
                (0, 6, 12),
                (2, 8, 14),
                (4, 10, 16),
                (0, 8, 16),
                (4, 8, 12),
            ] {
                let (x, y, z) = line;
                let x = x + player;
                let y = y + player;
                let z = z + player;
                if state[x] == state[y] && state[y] == state[z] && state[z] != 0 {
                    self.winners[game_idx] = (player + 1) as i8;
                    return;
                }
            }
        }
    }

    pub fn step(&mut self, actions: &[i8]) {
        let mut rng = rand::thread_rng();
        let batch_size = self.game_states.len();
        let mut available_moves = [0; 9];
        for i in 0..batch_size {
            if self.done[i] {
                self.reset_game(i);
            }
            let current_action = actions[i];

            let pos1 = (current_action as usize) * 2;
            let pos2 = (current_action as usize) * 2 + 1;
            if self.game_states[i][pos1] != 0 || self.game_states[i][pos2] != 0 {
                self.rewards[i] = -1;
                continue;
            }

            self.game_states[i][pos1] = 1;

            self.check_win(i);
            let num_moves_made = self.game_states[i].iter().filter(|&&x| x != 0).count();
            if num_moves_made == 9 {
                self.done[i] = true;
            }
            if self.winners[i] > 0 || self.done[i] {
                if self.winners[i] == 1 {
                    self.rewards[i] = 1;
                }
                for s in self.game_states[i].iter_mut() {
                    *s = 0;
                }
                continue;
            }

            let mut num_available_moves = 0;
            for j in 1..=9 {
                let idx1 = j * 2 - 2;
                let idx2 = j * 2 - 1;
                if self.game_states[i][idx1] == 0 && self.game_states[i][idx2] == 0 {
                    available_moves[j - 1] = j;
                    num_available_moves += 1;
                }
            }
            let opp_move = available_moves[rng.gen_range(0..num_available_moves)];
            let opp_idx = opp_move * 2 + 1;
            self.game_states[i][opp_idx] = 1;

            self.check_win(i);
            if self.winners[i] != 0 {
                self.rewards[i] = -1;
                self.done[i] = true;
                for s in self.game_states[i].iter_mut() {
                    *s = 0;
                }
            }
        }
    }
}

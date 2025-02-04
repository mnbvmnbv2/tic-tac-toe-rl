using Random

mutable struct TicTacToeEnv
    game_states::Matrix{Int16}
    rewards::Array{Int16}
    done::Array{Bool}
    winners::Array{Int16}
end

function TicTacToeEnv(batch_size::Int=1)
    game_states = zeros(Int16, batch_size, 18)
    rewards = zeros(Int16, batch_size)
    done = falses(batch_size)
    winners = zeros(Int16, batch_size)

    return TicTacToeEnv(game_states, rewards, done, winners)
end

function reset!(env::TicTacToeEnv)
    env.game_states .= 0
    env.rewards .= 0
    env.done .= false
    env.winners .= 0
end

function check_win!(env::TicTacToeEnv, game_idx::Int)
    # rows
    for i in 0:1
        for j in 0:2
            x = j * 6 + i + 1
            y = j * 6 + 2 + i + 1
            z = j * 6 + 4 + i + 1
            if (
                env.game_states[game_idx, x] == env.game_states[game_idx, y] &&
                env.game_states[game_idx, y] == env.game_states[game_idx, z] &&
                env.game_states[game_idx, z] != 0
            )
                env.winners[game_idx] = i + 1
                return
            end
        end
        # columns
        for j in 0:2
            x = j * 2 + i + 1
            y = j * 2 + 6 + i + 1
            z = j * 2 + 12 + i + 1
            if (
                env.game_states[game_idx, x] == env.game_states[game_idx, y] &&
                env.game_states[game_idx, y] == env.game_states[game_idx, z] &&
                env.game_states[game_idx, z] != 0
            )
                env.winners[game_idx] = i + 1
                return
            end
        end
        # diagonals
        x = i + 1
        y = i + 8 + 1
        z = i + 16 + 1
        if (
            env.game_states[game_idx, x] == env.game_states[game_idx, y] &&
            env.game_states[game_idx, y] == env.game_states[game_idx, z] &&
            env.game_states[game_idx, z] != 0
        )
            env.winners[game_idx] = i + 1
            return
        end
        x = i + 4 + 1
        y = i + 8 + 1
        z = i + 12 + 1
        if (
            env.game_states[game_idx, x] == env.game_states[game_idx, y] &&
            env.game_states[game_idx, y] == env.game_states[game_idx, z] &&
            env.game_states[game_idx, z] != 0
        )
            env.winners[game_idx] = i + 1
            return
        end
    end
end



function step!(env::TicTacToeEnv, actions::Vector{Int16})
    for i in 1:size(env.game_states, 1)
        if env.done[i]
            reset!(env)
            continue
        end
        current_action = actions[i]
        # if illegal move
        if env.game_states[i, current_action*2-1] != 0 || env.game_states[i, current_action*2] != 0
            env.rewards[i] = -1
            continue
        end
        # set action in game
        env.game_states[i, current_action*2-1] = 1
        num_moves_made = sum(env.game_states[i, :] .!= 0)
        # check if player won or game is full
        check_win!(env, i)
        if num_moves_made == 9
            env.done[i] = true
            continue
        end
        if env.winners[i] > 0 || env.done[i]
            if env.winners[i] == 1
                env.rewards[i] = 1
            end
            env.game_states[i, :] = 0
            continue
        end

        available_moves = Vector{Int}(undef, 9)
        num_available_moves = 0
        for j in 0:8
            pos1 = j * 2 + 1  # player 1's marker
            pos2 = j * 2 + 2  # player 2's marker
            if env.game_states[i, pos1] == 0 && env.game_states[i, pos2] == 0
                num_available_moves += 1
                available_moves[num_available_moves] = i
            end
        end
        opponent_action = available_moves[rand(1:num_available_moves)]
        println(opponent_action)
        opponent_idx = opponent_action * 2
        println(opponent_idx)
        env.game_states[i, opponent_idx] = 1
        check_win!(env, i)
        if env.winners[i] != 0
            env.rewards[i] = -1
            env.done[i] = true
            env.game_states[i, :] = 0
        end
    end
end

env = TicTacToeEnv(3)
check_win!(env, 1)
println(env.game_states)
step!(env, Int16[2, 3, 1])
println(env.game_states)
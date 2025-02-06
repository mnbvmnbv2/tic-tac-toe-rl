using Random, Flux

mutable struct TicTacToeEnv
    game_states::Matrix{Int8}
    rewards::Array{Int8}
    done::Array{Bool}
    winners::Array{Int8}
end

function TicTacToeEnv(batch_size::Int=1)
    game_states = zeros(Int8, batch_size, 18)
    rewards = zeros(Int8, batch_size)
    done = falses(batch_size)
    winners = zeros(Int8, batch_size)

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
        for j::Int in 0:2
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



function step!(env::TicTacToeEnv, actions::Vector{Int8})
    Threads.@threads for i::Int in 1:size(env.game_states, 1)
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
            env.game_states[i, :] .= 0
            continue
        end

        available_moves = Array{Int8}(undef, 9)
        num_available_moves = 0
        for j::Int in 1:9
            pos1 = j * 2 - 1  # player 1's marker
            pos2 = j * 2  # player 2's marker
            if env.game_states[i, pos1] == 0 && env.game_states[i, pos2] == 0
                num_available_moves += 1
                available_moves[num_available_moves] = j
            end
        end
        opponent_action = available_moves[rand(1:num_available_moves)]
        opponent_idx = opponent_action * 2
        env.game_states[i, opponent_idx] = 1
        check_win!(env, i)
        if env.winners[i] != 0
            env.rewards[i] = -1
            env.done[i] = true
            env.game_states[i, :] .= 0
        end
    end
end

# Benchmarking function: runs a fixed number of steps and times them.
function run_benchmark(batch_size::Int, num_steps::Int)
    env = TicTacToeEnv(batch_size)

    # Warm-up iterations (to compile and “warm-up” the JIT)
    for _ in 1:100
        actions = rand(Int8(1):Int8(9), batch_size)
        step!(env, actions)
    end

    start_time = time()
    for _ in 1:num_steps
        actions = rand(Int8(1):Int8(9), batch_size)
        step!(env, actions)
    end
    elapsed = time() - start_time
    steps_per_sec = num_steps * batch_size / elapsed
    println("Batch size: $batch_size -> $steps_per_sec steps per second")
end

# Run benchmarks for different batch sizes
# for batch in [1, 10, 100, 1000]
#     run_benchmark(batch, 10_000)
# end

# NN setup
model_cpu = Chain(
    Dense(18, 128, relu),
    Dense(128, 84, relu),
    Dense(84, 9)
)

# For GPU inference, move the model to the GPU.
# model_cuda = gpu(model_cpu)

function predict_cpu(env::TicTacToeEnv)
    # Transpose to get shape (features, batch)
    input = float.(env.game_states)'  # Convert to Float
    return model_cpu(input)
end

# function predict_cuda(env::TicTacToeEnv)
#     # Move the input to GPU and transpose
#     input = cu(float.(env.game_states)')
#     return model_cuda(input)
# end

function run_flux_benchmark(batch_size::Int, num_steps::Int; use_cuda::Bool=false)
    env = TicTacToeEnv(batch_size)
    # Warm-up steps: perform a few forward passes so that any compilation overhead is reduced.
    for _ in 1:10
        q_values = predict_cpu(env)
        actions = Int8.(vec(map(ci -> ci[1], argmax(q_values, dims=1))))
        step!(env, actions)
    end

    println("Running benchmark for Flux prediction (use_cuda=$(use_cuda))")
    start_time = time()
    for _ in 1:num_steps
        q_values = predict_cpu(env)
        actions = Int8.(vec(map(ci -> ci[1], argmax(q_values, dims=1))))
        step!(env, actions)
    end
    elapsed = time() - start_time
    steps_per_sec = num_steps * batch_size / elapsed
    println("Batch size: $batch_size -> $(round(steps_per_sec, digits=2)) steps per second")
end

run_flux_benchmark(1000, 1000, use_cuda=false)

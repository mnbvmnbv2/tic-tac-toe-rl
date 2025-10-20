using Flux
include("../env.jl")   # relative path from this file
using .Env

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
    env = getTicTacToeEnv(batch_size)
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
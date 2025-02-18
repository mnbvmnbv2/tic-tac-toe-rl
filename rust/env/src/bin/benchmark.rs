use env::TicTacToeEnv;
use rand::Rng;
use std::time::Instant;

fn run_benchmark(batch_size: usize, num_steps: usize) {
    let mut env = TicTacToeEnv::new(batch_size);
    let mut rng = rand::thread_rng();

    // Warm-up
    for _ in 0..100 {
        let actions: Vec<i8> = (0..batch_size).map(|_| rng.gen_range(0..9)).collect();
        env.step(&actions);
    }

    let start_time = Instant::now();
    for _ in 0..num_steps {
        let actions: Vec<i8> = (0..batch_size).map(|_| rng.gen_range(0..9)).collect();
        env.step(&actions);
    }
    let elapsed = start_time.elapsed().as_secs_f64();
    let steps_per_sec = (num_steps * batch_size) as f64 / elapsed;
    println!(
        "Rust - Batch size: {} -> {:.2} steps per second",
        batch_size, steps_per_sec
    );
}

fn main() {
    for &batch in &[1, 10, 100, 1000] {
        run_benchmark(batch, 10_000);
    }
}

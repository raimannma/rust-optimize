//! Async API Parameter Optimization Example
//!
//! This example shows how to use async/parallel optimization to tune
//! configuration parameters for a web service. Each evaluation simulates
//! an async operation (like deploying and load-testing a service).
//!
//! # Key Concepts Demonstrated
//!
//! - Async optimization with `optimize_parallel_with_sampler`
//! - Running multiple trials concurrently for faster optimization
//! - Boolean and categorical parameter types
//! - Measuring speedup from parallelism
//!
//! # When to Use Async Optimization
//!
//! Use async/parallel optimization when your objective function involves:
//! - Network requests (API calls, database queries)
//! - File I/O operations
//! - External service calls
//! - Any operation where you're waiting for I/O rather than computing
//!
//! With parallelism, you can evaluate multiple configurations simultaneously,
//! significantly reducing total optimization time.
//!
//! Run with: `cargo run --example async_api_optimization --features async`

use std::time::{Duration, Instant};

use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, ParamValue, Study, Trial};

// ============================================================================
// Configuration: Service parameters we want to tune
// ============================================================================

/// Configuration for a web service.
///
/// In a real application, these parameters would control:
/// - Memory allocation (cache sizes)
/// - Connection management (pool sizes, timeouts)
/// - Request handling (batching, compression)
/// - Protocol options (HTTP version, load balancing)
struct ServiceConfig {
    cache_size_mb: i64,
    connection_pool_size: i64,
    request_timeout_ms: i64,
    retry_count: i64,
    batch_size: i64,
    compression_level: i64,
    use_http2: bool,
    load_balancing: String,
}

// ============================================================================
// Objective Function: Evaluate a service configuration
// ============================================================================

/// Simulates deploying and load-testing a service configuration.
///
/// In a real scenario, this function might:
/// 1. Deploy the configuration to a staging environment
/// 2. Run load tests against the service
/// 3. Collect metrics (latency, throughput, error rate)
/// 4. Return a composite score
///
/// The async sleep simulates the I/O time of these operations.
/// This is where parallel execution helps - while one trial is waiting
/// for I/O, other trials can run.
#[allow(clippy::too_many_arguments)]
async fn evaluate_service(config: &ServiceConfig) -> f64 {
    // Simulate async I/O (deployment, load testing, metric collection)
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Calculate a score based on how close we are to optimal values
    // Lower score = better configuration
    let mut score = 0.0;

    // Cache size: too small = cache misses, too large = wasted memory
    // Optimal around 512MB
    let cache_optimal = 512.0;
    score += ((config.cache_size_mb as f64 - cache_optimal) / 256.0).powi(2);

    // Connection pool: too small = contention, too large = resource waste
    // Optimal around 100
    let pool_optimal = 100.0;
    score += ((config.connection_pool_size as f64 - pool_optimal) / 50.0).powi(2);

    // Timeout: too short = false failures, too long = slow recovery
    // Optimal around 5000ms
    let timeout_optimal = 5000.0;
    score += ((config.request_timeout_ms as f64 - timeout_optimal) / 2000.0).powi(2);

    // Retries: too few = fragile, too many = amplifies failures
    // Optimal around 3
    let retry_optimal = 3.0;
    score += ((config.retry_count as f64 - retry_optimal) / 2.0).powi(2);

    // Batch size: trade-off between latency and throughput
    // Optimal around 64
    let batch_optimal = 64.0;
    score += ((config.batch_size as f64 - batch_optimal) / 32.0).powi(2);

    // Compression level: trade-off between CPU and bandwidth
    // Optimal around 6
    let compression_optimal = 6.0;
    score += ((config.compression_level as f64 - compression_optimal) / 3.0).powi(2);

    // HTTP/2 is generally better for our use case
    if !config.use_http2 {
        score += 0.5;
    }

    // Load balancing strategy affects performance
    score += match config.load_balancing.as_str() {
        "round_robin" => 0.0,       // Best for our use case
        "least_connections" => 0.1, // Good alternative
        "ip_hash" => 0.2,           // OK for session affinity
        "random" => 0.3,            // Not ideal
        _ => 1.0,
    };

    // Add noise to simulate real-world variability
    let noise = (config.cache_size_mb as f64 * 0.1).sin() * 0.05;

    score + noise
}

/// The async objective function for each trial.
///
/// For async optimization, the objective function must:
/// 1. Take ownership of the Trial (not a mutable reference)
/// 2. Return a Future
/// 3. Return both the Trial and the result value as a tuple
///
/// This ownership pattern allows the trial to be used across await points.
async fn objective(mut trial: Trial) -> optimizer::Result<(Trial, f64)> {
    // Sample configuration parameters

    // Stepped integers: only sample multiples of the step value
    let cache_size_mb = trial.suggest_int_step("cache_size_mb", 64, 1024, 64)?;
    let connection_pool_size = trial.suggest_int_step("connection_pool_size", 10, 200, 10)?;
    let request_timeout_ms = trial.suggest_int_step("request_timeout_ms", 1000, 10000, 500)?;

    // Regular integer
    let retry_count = trial.suggest_int("retry_count", 0, 5)?;

    // Log-scale integer: good for parameters like batch sizes
    // that might vary from 1 to 256
    let batch_size = trial.suggest_int_log("batch_size", 1, 256)?;

    // Regular integer for compression level
    let compression_level = trial.suggest_int("compression_level", 0, 9)?;

    // Boolean: internally uses categorical with [false, true]
    let use_http2 = trial.suggest_bool("use_http2")?;

    // Categorical: choose from a list of options
    let load_balancing = trial.suggest_categorical(
        "load_balancing",
        &["round_robin", "least_connections", "random", "ip_hash"],
    )?;

    // Build configuration
    let config = ServiceConfig {
        cache_size_mb,
        connection_pool_size,
        request_timeout_ms,
        retry_count,
        batch_size,
        compression_level,
        use_http2,
        load_balancing: load_balancing.to_string(),
    };

    // Evaluate (this is the async part)
    let score = evaluate_service(&config).await;

    // Return both the trial and the score
    Ok((trial, score))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Formats a parameter value for display.
fn format_param(name: &str, value: &ParamValue) -> String {
    match (name, value) {
        (_, ParamValue::Float(v)) => format!("{v:.4}"),
        (_, ParamValue::Int(v)) => format!("{v}"),
        ("use_http2", ParamValue::Categorical(idx)) => {
            if *idx == 1 { "true" } else { "false" }.to_string()
        }
        ("load_balancing", ParamValue::Categorical(idx)) => {
            ["round_robin", "least_connections", "random", "ip_hash"]
                .get(*idx)
                .unwrap_or(&"unknown")
                .to_string()
        }
        (_, ParamValue::Categorical(idx)) => format!("category_{idx}"),
    }
}

/// Prints the results of the optimization.
fn print_results(study: &Study<f64>, elapsed: Duration, n_trials: usize) {
    println!("\n{}", "=".repeat(60));
    println!("\nOptimization completed!");
    println!("Total trials: {}", study.n_trials());
    println!("Time elapsed: {elapsed:.2?}");

    // Calculate speedup from parallelism
    // Each trial takes ~50ms, so sequential would take n_trials * 50ms
    let sequential_time = n_trials as f64 * 0.050;
    let actual_time = elapsed.as_secs_f64();
    println!(
        "Effective parallelism: {:.1}x speedup",
        sequential_time / actual_time
    );
}

/// Prints the best configuration found.
fn print_best_config(study: &Study<f64>) -> optimizer::Result<()> {
    let best = study.best_trial()?;

    println!("\nBest configuration found:");
    println!("  Score: {:.6}", best.value);
    println!("\n  Parameters:");

    // Print parameters in a logical order
    let param_order = [
        "cache_size_mb",
        "connection_pool_size",
        "request_timeout_ms",
        "retry_count",
        "batch_size",
        "compression_level",
        "use_http2",
        "load_balancing",
    ];

    for name in param_order {
        if let Some(value) = best.params.get(name) {
            let display = format_param(name, value);
            println!("    {name}: {display}");
        }
    }

    Ok(())
}

/// Prints the top N trials.
fn print_top_trials(study: &Study<f64>, n: usize) {
    println!("\nTop {n} trials:");

    let mut trials = study.trials();
    trials.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

    for (i, trial) in trials.iter().take(n).enumerate() {
        println!(
            "  {}. Trial #{}: score = {:.6}",
            i + 1,
            trial.id,
            trial.value
        );
    }
}

// ============================================================================
// Main: Set up and run the async optimization
// ============================================================================

#[tokio::main]
async fn main() -> optimizer::Result<()> {
    println!("=== Async API Parameter Optimization Example ===\n");

    // Step 1: Create a TPE sampler
    let sampler = TpeSampler::builder()
        .n_startup_trials(8)
        .gamma(0.2)
        .seed(123)
        .build()
        .expect("Failed to build TPE sampler");

    // Step 2: Create a study to minimize the score
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    // Step 3: Configure optimization
    let n_trials = 40;
    let concurrency = 4; // Run 4 trials in parallel

    println!("Starting parallel optimization with {concurrency} concurrent evaluations...\n");

    let start = Instant::now();

    // Step 4: Run parallel async optimization
    //
    // optimize_parallel_with_sampler:
    // - Runs up to `concurrency` trials simultaneously
    // - Each trial calls the objective function
    // - Uses a semaphore to limit concurrent evaluations
    // - Collects results as trials complete
    //
    // The "_with_sampler" suffix means the TPE sampler gets access to
    // trial history for informed sampling.
    study
        .optimize_parallel_with_sampler(n_trials, concurrency, objective)
        .await?;

    let elapsed = start.elapsed();

    // Step 5: Print results
    print_results(&study, elapsed, n_trials);
    print_best_config(&study)?;
    print_top_trials(&study, 5);

    Ok(())
}

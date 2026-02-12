//! Advanced Features Example
//!
//! This example demonstrates four advanced capabilities of the optimizer crate:
//!
//! 1. **Async parallel optimization** — evaluate multiple trials concurrently
//! 2. **Journal storage** — persist trials to disk and resume studies later
//! 3. **Ask-and-tell interface** — decouple sampling from evaluation
//! 4. **Multi-objective optimization** — optimize competing objectives simultaneously
//!
//! Run with: `cargo run --example advanced_features --features "async,journal"`

use std::time::Instant;

use optimizer::multi_objective::MultiObjectiveStudy;
use optimizer::prelude::*;

// ============================================================================
// Section 1: Async Parallel Optimization
// ============================================================================

/// Runs multiple trials concurrently using tokio, reducing wall-clock time
/// when the objective function involves I/O or other async work.
async fn async_parallel_optimization() -> optimizer::Result<()> {
    println!("=== Section 1: Async Parallel Optimization ===\n");

    let sampler = TpeSampler::builder()
        .n_startup_trials(5)
        .seed(42)
        .build()
        .expect("Failed to build TPE sampler");

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    let n_trials = 30;
    let concurrency = 4;

    println!("Running {n_trials} trials with {concurrency} concurrent workers...");
    let start = Instant::now();

    // optimize_parallel spawns up to `concurrency` trials at once.
    // The closure must take ownership of Trial and return (Trial, value).
    study
        .optimize_parallel(n_trials, concurrency, {
            let x = x.clone();
            let y = y.clone();
            move |mut trial| {
                let x = x.clone();
                let y = y.clone();
                async move {
                    let xv = x.suggest(&mut trial)?;
                    let yv = y.suggest(&mut trial)?;

                    // Simulate async I/O (e.g., calling an external service)
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

                    // Sphere function: minimum at origin
                    let value = xv * xv + yv * yv;
                    Ok::<_, optimizer::Error>((trial, value))
                }
            }
        })
        .await?;

    let elapsed = start.elapsed();
    let best = study.best_trial()?;

    println!(
        "Completed in {elapsed:.2?} (vs ~{:.0?} sequential)",
        std::time::Duration::from_millis(10 * n_trials as u64)
    );
    println!(
        "Best: f({:.3}, {:.3}) = {:.6}\n",
        best.get(&x).unwrap(),
        best.get(&y).unwrap(),
        best.value
    );

    Ok(())
}

// ============================================================================
// Section 2: Journal Storage
// ============================================================================

/// Persists trials to a JSONL file so that a study can be resumed later.
/// Useful for long-running experiments or crash recovery.
fn journal_storage_demo() -> optimizer::Result<()> {
    println!("=== Section 2: Journal Storage ===\n");

    let path = std::env::temp_dir().join("optimizer_advanced_example.jsonl");

    // Clean up from any previous run
    let _ = std::fs::remove_file(&path);

    let x = FloatParam::new(-5.0, 5.0).name("x");

    // --- First run: optimize 20 trials and persist to disk ---
    {
        let storage = JournalStorage::<f64>::new(&path);
        let study: Study<f64> = Study::builder()
            .minimize()
            .sampler(TpeSampler::new())
            .storage(storage)
            .build();

        study.optimize(20, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })?;

        println!(
            "First run: {} trials saved to {}",
            study.n_trials(),
            path.display()
        );
    }

    // --- Second run: resume from the journal file ---
    {
        // JournalStorage::open loads existing trials from disk
        let storage = JournalStorage::<f64>::open(&path)?;
        let study: Study<f64> = Study::builder()
            .minimize()
            .sampler(TpeSampler::new())
            .storage(storage)
            .build();

        // The sampler sees the prior 20 trials, so it starts informed
        let before = study.n_trials();
        study.optimize(10, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })?;

        let best = study.best_trial()?;
        println!(
            "Resumed: {} → {} trials, best f({:.4}) = {:.6}",
            before,
            study.n_trials(),
            best.get(&x).unwrap(),
            best.value
        );
    }

    // Clean up the temporary file
    let _ = std::fs::remove_file(&path);

    println!();
    Ok(())
}

// ============================================================================
// Section 3: Ask-and-Tell Interface
// ============================================================================

/// Decouples trial creation from evaluation. Useful when:
/// - Evaluations happen outside the optimizer (e.g., in a separate process)
/// - You want to batch evaluations before reporting results
/// - You need custom scheduling logic
fn ask_and_tell_demo() -> optimizer::Result<()> {
    println!("=== Section 3: Ask-and-Tell Interface ===\n");

    let study: Study<f64> = Study::new(Direction::Minimize);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    // Ask for a batch of trials, evaluate externally, then tell results
    for batch in 0..3 {
        let batch_size = 5;
        let mut trials = Vec::with_capacity(batch_size);

        // ask() creates trials with sampled parameters
        for _ in 0..batch_size {
            let mut trial = study.ask();
            let xv = x.suggest(&mut trial)?;
            let yv = y.suggest(&mut trial)?;

            // Store values alongside the trial for later evaluation
            trials.push((trial, xv, yv));
        }

        // Evaluate the batch (could be sent to workers, GPUs, etc.)
        for (trial, xv, yv) in trials {
            let value = xv * xv + yv * yv;
            // tell() reports the result back to the study
            study.tell(trial, Ok::<_, &str>(value));
        }

        println!(
            "Batch {}: evaluated {} trials (total: {})",
            batch + 1,
            batch_size,
            study.n_trials()
        );
    }

    let best = study.best_trial()?;
    println!(
        "Best: f({:.3}, {:.3}) = {:.6}\n",
        best.get(&x).unwrap(),
        best.get(&y).unwrap(),
        best.value
    );

    Ok(())
}

// ============================================================================
// Section 4: Multi-Objective Optimization
// ============================================================================

/// Optimizes two competing objectives simultaneously.
/// Returns the Pareto front — the set of solutions where no objective can
/// be improved without worsening the other.
fn multi_objective_demo() -> optimizer::Result<()> {
    println!("=== Section 4: Multi-Objective Optimization ===\n");

    // Two objectives, both minimized
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);

    let x = FloatParam::new(0.0, 1.0).name("x");

    // Classic bi-objective problem: f1(x) = x², f2(x) = (x - 1)²
    // The Pareto front is the curve where improving f1 worsens f2 and vice versa.
    study.optimize(50, |trial| {
        let xv = x.suggest(trial)?;
        let f1 = xv * xv;
        let f2 = (xv - 1.0) * (xv - 1.0);
        Ok::<_, optimizer::Error>(vec![f1, f2])
    })?;

    let front = study.pareto_front();
    println!(
        "Ran {} trials, Pareto front has {} solutions:",
        study.n_trials(),
        front.len()
    );

    // Show a few Pareto-optimal trade-offs
    let mut sorted_front = front.clone();
    sorted_front.sort_by(|a, b| a.values[0].partial_cmp(&b.values[0]).unwrap());

    for (i, trial) in sorted_front.iter().take(5).enumerate() {
        println!(
            "  {}: x={:.3}, f1={:.4}, f2={:.4}",
            i + 1,
            trial.get(&x).unwrap(),
            trial.values[0],
            trial.values[1]
        );
    }
    if sorted_front.len() > 5 {
        println!("  ... and {} more", sorted_front.len() - 5);
    }

    println!();
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> optimizer::Result<()> {
    async_parallel_optimization().await?;
    journal_storage_demo()?;
    ask_and_tell_demo()?;
    multi_objective_demo()?;

    println!("All sections completed successfully!");
    Ok(())
}

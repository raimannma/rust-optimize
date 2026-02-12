//! Pruning and early-stopping example — demonstrates trial pruning with `MedianPruner`
//! and early stopping via `optimize_with_callback`.
//!
//! Simulates a training loop where each trial trains for multiple "epochs". The pruner
//! stops unpromising trials early, and a callback halts the entire study once a target
//! loss is reached.
//!
//! Run with: `cargo run --example pruning_and_callbacks`

use std::ops::ControlFlow;

use optimizer::TrialState;
use optimizer::prelude::*;

fn main() -> optimizer::Result<()> {
    let n_trials: usize = 30;
    let n_epochs: u64 = 20;
    let target_loss = 0.15;

    // Build a study with a seeded random sampler and MedianPruner.
    // MedianPruner compares each trial's intermediate value against the median of
    // completed trials at the same step — trials performing below median are pruned.
    let study: Study<f64> = Study::builder()
        .minimize()
        .sampler(RandomSampler::with_seed(42))
        .pruner(
            MedianPruner::new(Direction::Minimize)
                .n_warmup_steps(3) // let every trial run at least 3 epochs before pruning
                .n_min_trials(3), // need 3 completed trials before pruning kicks in
        )
        .build();

    let learning_rate = FloatParam::new(1e-4, 1.0).name("learning_rate");
    let momentum = FloatParam::new(0.0, 0.99).name("momentum");

    // Use optimize_with_callback to get both pruning AND early stopping.
    // The callback fires after each completed (or pruned) trial and can halt the study.
    study.optimize_with_callback(
        n_trials,
        // --- Objective function: simulated training loop with pruning ---
        |trial| {
            let lr = learning_rate.suggest(trial)?;
            let mom = momentum.suggest(trial)?;

            // Simulate training for n_epochs, reporting intermediate loss each epoch.
            // Good hyperparameters (lr ≈ 0.01, momentum ≈ 0.8) converge to low loss;
            // bad combos plateau high — giving the pruner something to cut.
            let mut loss = 1.0;
            for epoch in 0..n_epochs {
                let lr_penalty = (lr.log10() - 0.01_f64.log10()).powi(2); // 0 at lr=0.01
                let mom_penalty = (mom - 0.8).powi(2); // 0 at momentum=0.8
                let base_loss = 0.02 + 0.05 * lr_penalty + 1.5 * mom_penalty;
                let progress = (epoch as f64 + 1.0) / n_epochs as f64;
                // Loss decays from 1.0 toward base_loss over epochs.
                loss = base_loss + (1.0 - base_loss) * (-3.5 * progress).exp();

                // Report the intermediate value so the pruner can evaluate this trial.
                trial.report(epoch, loss);

                // Check whether the pruner recommends stopping this trial early.
                if trial.should_prune() {
                    // Signal that this trial was pruned — the study records it as Pruned.
                    Err(TrialPruned)?;
                }
            }

            Ok::<_, Error>(loss)
        },
        // --- Callback: early stopping when we hit the target ---
        |study, completed_trial| {
            let n_complete = study.n_trials();
            let n_pruned = study
                .trials()
                .iter()
                .filter(|t| t.state == TrialState::Pruned)
                .count();

            match completed_trial.state {
                TrialState::Pruned => {
                    println!(
                        "  Trial {:>3} PRUNED at epoch {} (loss = {:.4}) \
                         [{n_complete} done, {n_pruned} pruned]",
                        completed_trial.id,
                        completed_trial.intermediate_values.len(),
                        completed_trial
                            .intermediate_values
                            .last()
                            .map_or(f64::NAN, |v| v.1),
                    );
                }
                TrialState::Complete => {
                    println!(
                        "  Trial {:>3} complete: loss = {:.4} \
                         [{n_complete} done, {n_pruned} pruned]",
                        completed_trial.id, completed_trial.value,
                    );
                }
                _ => {}
            }

            // Stop the entire study once we find a good enough result.
            if completed_trial.state == TrialState::Complete && completed_trial.value < target_loss
            {
                println!("\n  Early stopping: reached target loss {target_loss}!");
                return ControlFlow::Break(());
            }

            ControlFlow::Continue(())
        },
    )?;

    // --- Results ---
    let best = study.best_trial().expect("at least one completed trial");
    let total = study.n_trials();
    let pruned = study
        .trials()
        .iter()
        .filter(|t| t.state == TrialState::Pruned)
        .count();

    println!("\n--- Results ---");
    println!("  Total trials : {total}");
    println!("  Pruned       : {pruned}");
    println!("  Completed    : {}", total - pruned);
    println!("  Best trial #{}: loss = {:.6}", best.id, best.value);
    println!(
        "    learning_rate = {:.6}",
        best.get(&learning_rate).unwrap()
    );
    println!("    momentum      = {:.4}", best.get(&momentum).unwrap());

    Ok(())
}

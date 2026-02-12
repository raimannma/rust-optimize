use optimizer::prelude::*;
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::sobol::SobolSampler;

#[test]
fn sphere_function() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(xv * xv + yv * yv)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 100.0,
        "sphere best value should be < 100.0, got {}",
        best.value
    );
}

#[test]
fn bounds_respected() {
    let sampler = SobolSampler::with_seed(123);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-2.0, 3.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(xv + yv)
        })
        .unwrap();

    for trial in study.trials() {
        let xv: f64 = trial.get(&x).unwrap();
        let yv: f64 = trial.get(&y).unwrap();
        assert!((-2.0..=3.0).contains(&xv), "x = {xv} out of bounds [-2, 3]");
        assert!((0.0..=10.0).contains(&yv), "y = {yv} out of bounds [0, 10]");
    }
}

#[test]
fn integer_params() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let n = IntParam::new(1, 20).name("n");

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let nv = n.suggest(trial)?;
            Ok::<_, Error>(((nv - 10) * (nv - 10)) as f64)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    let best_n: i64 = best.get(&n).unwrap();
    assert!(
        (1..=20).contains(&best_n),
        "integer value {best_n} out of bounds"
    );
    assert!(
        best.value < 10.0,
        "integer optimization should find a good value, got {}",
        best.value
    );
}

#[test]
fn log_scale_params() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let lr = FloatParam::new(1e-5, 1.0).log_scale().name("lr");

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let lrv = lr.suggest(trial)?;
            Ok::<_, Error>((lrv.ln() - 0.01_f64.ln()).powi(2))
        })
        .unwrap();

    for trial in study.trials() {
        let lrv: f64 = trial.get(&lr).unwrap();
        assert!(
            (1e-5..=1.0).contains(&lrv),
            "log-scale value {lrv} out of bounds"
        );
    }
}

#[test]
fn categorical_params() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let cat = CategoricalParam::new(vec!["a", "b", "c"]).name("cat");

    study
        .optimize(50, |trial: &mut optimizer::Trial| {
            let cv = cat.suggest(trial)?;
            let val = match cv {
                "a" => 0.0,
                "b" => 1.0,
                _ => 2.0,
            };
            Ok::<_, Error>(val)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 2.0,
        "categorical optimization should find a good value, got {}",
        best.value
    );
}

#[test]
fn mixed_params() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let n = IntParam::new(1, 10).name("n");
    let cat = CategoricalParam::new(vec!["a", "b", "c"]).name("cat");

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let nv = n.suggest(trial)?;
            let cv = cat.suggest(trial)?;
            let penalty = match cv {
                "a" => 0.0,
                "b" => 1.0,
                _ => 2.0,
            };
            Ok::<_, Error>(xv * xv + nv as f64 + penalty)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 20.0,
        "mixed-param optimization should find a reasonable value, got {}",
        best.value
    );
}

#[test]
fn single_dimension() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-10.0, 10.0).name("x");

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, Error>((xv - 3.0).powi(2))
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 5.0,
        "1-D optimization should find a decent value, got {}",
        best.value
    );
}

#[test]
fn many_dimensions() {
    let sampler = SobolSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let params: Vec<FloatParam> = (0..8)
        .map(|i| FloatParam::new(-5.0, 5.0).name(format!("x{i}")))
        .collect();

    study
        .optimize(200, |trial: &mut optimizer::Trial| {
            let mut sum = 0.0;
            for p in &params {
                let v = p.suggest(trial)?;
                sum += v * v;
            }
            Ok::<_, Error>(sum)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    // With 8 dimensions and 200 quasi-random trials the best won't be amazing,
    // but it should be noticeably below the worst case (8 * 25 = 200).
    assert!(
        best.value < 150.0,
        "8-D optimization should find something reasonable, got {}",
        best.value
    );
}

#[test]
fn seeded_reproducibility() {
    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    let run = |seed: u64| {
        let sampler = SobolSampler::with_seed(seed);
        let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
        study
            .optimize(50, |trial: &mut optimizer::Trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, Error>(xv * xv + yv * yv)
            })
            .unwrap();
        study.trials().iter().map(|t| t.value).collect::<Vec<_>>()
    };

    let results1 = run(42);
    let results2 = run(42);
    assert_eq!(results1, results2, "same seed should produce same results");
}

#[test]
fn different_seeds_different_results() {
    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    let run = |seed: u64| {
        let sampler = SobolSampler::with_seed(seed);
        let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
        study
            .optimize(20, |trial: &mut optimizer::Trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, Error>(xv * xv + yv * yv)
            })
            .unwrap();
        study.trials().iter().map(|t| t.value).collect::<Vec<_>>()
    };

    let results1 = run(42);
    let results2 = run(99);
    assert_ne!(
        results1, results2,
        "different seeds should produce different results"
    );
}

#[test]
fn better_coverage_than_random() {
    let n_trials = 30;
    let n_bins = 10;

    let x = FloatParam::new(0.0, 1.0).name("x");

    // Count bins filled by Sobol.
    let sobol_study: Study<f64> =
        Study::with_sampler(Direction::Minimize, SobolSampler::with_seed(0));
    sobol_study
        .optimize(n_trials, |trial: &mut optimizer::Trial| {
            let v = x.suggest(trial)?;
            Ok::<_, Error>(v)
        })
        .unwrap();

    let mut sobol_bins = vec![0u32; n_bins];
    for trial in sobol_study.trials() {
        let v: f64 = trial.get(&x).unwrap();
        let bin = ((v * n_bins as f64).floor() as usize).min(n_bins - 1);
        sobol_bins[bin] += 1;
    }
    let sobol_filled = sobol_bins.iter().filter(|&&c| c > 0).count();

    // Count bins filled by Random.
    let random_study: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(0));
    random_study
        .optimize(n_trials, |trial: &mut optimizer::Trial| {
            let v = x.suggest(trial)?;
            Ok::<_, Error>(v)
        })
        .unwrap();

    let mut random_bins = vec![0u32; n_bins];
    for trial in random_study.trials() {
        let v: f64 = trial.get(&x).unwrap();
        let bin = ((v * n_bins as f64).floor() as usize).min(n_bins - 1);
        random_bins[bin] += 1;
    }
    let random_filled = random_bins.iter().filter(|&&c| c > 0).count();

    assert!(
        sobol_filled >= random_filled,
        "Sobol should fill at least as many bins as random: sobol={sobol_filled}, random={random_filled}"
    );
    // Sobol with 30 samples in 10 bins should fill all or nearly all bins.
    assert!(
        sobol_filled >= 9,
        "Sobol should fill at least 9/10 bins, got {sobol_filled}: {sobol_bins:?}"
    );
}

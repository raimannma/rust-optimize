use optimizer::parameter::{CategoricalParam, FloatParam, IntParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Study};

#[test]
fn known_perfect_correlation() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 100.0).name("x");

    // Objective = x, so perfect correlation.
    for _ in 0..30 {
        let mut trial = study.ask();
        let xv = x.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(xv));
    }

    let importance = study.param_importance();
    assert_eq!(importance.len(), 1);
    assert_eq!(importance[0].0, "x");
    assert!(
        (importance[0].1 - 1.0).abs() < 1e-10,
        "single param should be 1.0"
    );
}

#[test]
fn no_effect_parameter() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 100.0).name("x");
    let noise = FloatParam::new(0.0, 100.0).name("noise");

    // Objective depends only on x; noise is unused in objective.
    for _ in 0..50 {
        let mut trial = study.ask();
        let xv = x.suggest(&mut trial).unwrap();
        let _nv = noise.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(xv));
    }

    let importance = study.param_importance();
    assert_eq!(importance.len(), 2);
    // x should have much higher importance than noise.
    let x_score = importance.iter().find(|(l, _)| l == "x").unwrap().1;
    let noise_score = importance.iter().find(|(l, _)| l == "noise").unwrap().1;
    assert!(
        x_score > noise_score,
        "x ({x_score}) should outrank noise ({noise_score})"
    );
    // x should dominate
    assert!(x_score > 0.7, "x importance {x_score} should be dominant");
}

#[test]
fn multiple_parameters_varying_importance() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    // Objective = 10*x + 0.01*y → x should be far more important.
    for _ in 0..50 {
        let mut trial = study.ask();
        let xv = x.suggest(&mut trial).unwrap();
        let yv = y.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(10.0 * xv + 0.01 * yv));
    }

    let importance = study.param_importance();
    assert_eq!(importance.len(), 2);
    assert_eq!(importance[0].0, "x", "x should rank first");
    assert!(importance[0].1 > importance[1].1);
}

#[test]
fn fewer_than_two_trials_returns_empty() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // 0 trials
    assert!(study.param_importance().is_empty());

    // 1 trial
    let x = FloatParam::new(0.0, 1.0).name("x");
    let mut trial = study.ask();
    let xv = x.suggest(&mut trial).unwrap();
    study.tell(trial, Ok::<_, &str>(xv));
    assert!(study.param_importance().is_empty());
}

#[test]
fn int_parameter() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let n = IntParam::new(1, 100).name("n");

    for _ in 0..30 {
        let mut trial = study.ask();
        let nv = n.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(nv as f64));
    }

    let importance = study.param_importance();
    assert_eq!(importance.len(), 1);
    assert_eq!(importance[0].0, "n");
    assert!(importance[0].1 > 0.9);
}

#[test]
fn categorical_parameter() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let cat = CategoricalParam::new(vec!["a", "b", "c"]).name("cat");
    let x = FloatParam::new(0.0, 100.0).name("x");

    // Objective depends only on x; categorical is random noise.
    for _ in 0..50 {
        let mut trial = study.ask();
        let _c = cat.suggest(&mut trial).unwrap();
        let xv = x.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(xv));
    }

    let importance = study.param_importance();
    assert_eq!(importance.len(), 2);
    let x_score = importance.iter().find(|(l, _)| l == "x").unwrap().1;
    assert!(x_score > 0.5, "x should dominate over categorical noise");
}

#[test]
fn normalization_sums_to_one() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");
    let z = FloatParam::new(0.0, 10.0).name("z");

    for _ in 0..50 {
        let mut trial = study.ask();
        let xv = x.suggest(&mut trial).unwrap();
        let yv = y.suggest(&mut trial).unwrap();
        let zv = z.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(xv + 0.5 * yv + 0.1 * zv));
    }

    let importance = study.param_importance();
    let sum: f64 = importance.iter().map(|(_, s)| *s).sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "scores should sum to 1.0, got {sum}"
    );
}

#[test]
fn label_when_unnamed_uses_debug() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    // No .name() call → label defaults to Debug representation.
    let x = FloatParam::new(0.0, 10.0);

    for _ in 0..10 {
        let mut trial = study.ask();
        let xv = x.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>(xv));
    }

    let importance = study.param_importance();
    assert_eq!(importance.len(), 1);
    assert!(
        importance[0].0.starts_with("FloatParam"),
        "expected Debug label, got {:?}",
        importance[0].0
    );
}

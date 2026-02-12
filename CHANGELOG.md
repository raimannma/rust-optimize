# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-02-12

### Fixed

- Fix journal file handling to support both reading and writing modes

## [0.9.0] - 2026-02-12

### Added

- Storage trait abstraction with pluggable backends
- JSONL journal storage backend for persistent studies (behind `journal` feature)
- Differential Evolution sampler
- Gaussian Process sampler with Expected Improvement acquisition function
- NSGA-III sampler for many-objective optimization with reference-point decomposition
- MOEA/D sampler for decomposition-based multi-objective optimization
- `StudyBuilder` for fluent study construction

### Changed

- Move `next_trial_id` counter from `Study` into `Storage` trait
- Replace `rand 0.10` with `fastrand 2.3` for simpler, faster random number generation
- Remove `visualization` and `fanova` feature flags (now always available)
- Simplify CI test commands by removing feature matrix

### Fixed

- Seed importance tests to eliminate flakiness
- Use atomic counter for temp paths in journal tests

### Removed

- Reverted SQLite storage backend (may return in a future release)

## [0.8.1] - 2026-02-11

### Fixed

- Resolve broken rustdoc intra-doc links for `Error::NoCompletedTrials`

## [0.8.0] - 2026-02-11

### Added

- Multi-objective optimization with NSGA-II
- Multi-Objective TPE (MOTPE) sampler
- Pareto front analysis utilities
- CSV and JSON data export for visualization
- HTML visualization reports with Plotly.js charts
- fANOVA (functional ANOVA) parameter importance via random forest

### Fixed

- Rename variable to pass typos check

## [0.7.2] - 2026-02-11

### Changed

- Update `nalgebra` dependency to version 0.34
- Add advisory ignore for unmaintained transitive dependency

## [0.7.1] - 2026-02-11

### Changed

- Update `tracing` dependency to version 0.1.29

## [0.7.0] - 2026-02-11

### Added

- CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler behind `cma-es` feature flag
- Sobol quasi-random sampler behind `sobol` feature flag
- BOHB (Bayesian Optimization with HyperBand) sampler for budget-aware optimization
- Parameter importance analysis via Spearman rank correlation
- Constraint handling with feasibility-aware trial ranking
- `optimize_with_retries()` for automatic retry of failed trials
- `optimize_with_checkpoint()` with atomic save writes for crash recovery
- `Study::summary()` and `Display` impl for study overview
- `IntoIterator` for `&Study` and `iter()` method
- Tracing integration behind `tracing` feature flag
- Serde serialization support behind `serde` feature flag
- Benchmark suite with criterion and standard test functions

## [0.6.0] - 2026-02-11

### Added

- Pruning system with `Pruner` trait and `NopPruner` default implementation
- Intermediate value reporting on trials for pruner integration
- `TrialPruned` error variant and `Pruned` trial state
- `ThresholdPruner` for fixed-bound trial pruning
- `MedianPruner` for statistics-based trial pruning
- `PercentilePruner` for configurable percentile-based trial pruning
- `PatientPruner` for patience-based trial pruning
- `SuccessiveHalvingPruner` for SHA-based trial pruning
- `HyperbandPruner` for multi-bracket trial pruning
- `WilcoxonPruner` for statistics-based trial pruning
- `Study::minimize()` and `Study::maximize()` constructor shortcuts
- `From<RangeInclusive>` for `FloatParam` and `IntParam`
- Timeout-based optimization with `optimize_until`
- `Study::top_trials(n)` for retrieving the best N trials
- Ask-and-tell interface with `ask()` and `tell()` methods
- Trial user attributes for logging and analysis
- `enqueue_trial()` for pre-specified parameter evaluation

## [0.5.1] - 2026-02-10

### Changed

- Update random number generation to use `rand::make_rng()` and upgrade `rand` to version 0.10

## [0.5.0] - 2026-02-06

### Added

- Typed parameter API with `FloatParam`, `IntParam`, `CategoricalParam`, `BoolParam`, and `EnumParam`
- `#[derive(Categorical)]` proc macro for deriving categorical parameters from enums (behind `derive` feature)
- `.name()` builder method on all parameter types for custom labels
- `CompletedTrial::get()` for typed parameter access
- `Display` impl on `ParamValue`
- Prelude module at `optimizer::prelude::*`

### Changed

- Reorganize sampler module structure and update imports
- Remove `log` dependency

## [0.4.0] - 2026-02-02

### Added

- Multivariate TPE sampler for correlated parameter search
- Gamma strategies for TPE sampler (linear, sqrt, fixed) with examples
- Example: async API parameter optimization
- Example: ML hyperparameter tuning

### Fixed

- Handle end value in `suggest` method to avoid panic on underflow

## [0.3.1] - 2026-01-31

### Added

- Grid search sampler for exhaustive parameter exploration
- `suggest_bool()` method for boolean parameter suggestion
- `SuggestableRange` trait and `suggest_range()` method for parameter suggestion from ranges
- Documentation, keywords, categories, and readme fields in `Cargo.toml`

### Changed

- Replace `TpeError` with unified `Error` type across the library
- Add permissions for read access to contents in CI and scheduled workflows

## [0.3.0] - 2026-01-30

### Added

- Cross-target compilation checks in CI

### Changed

- Internal code refactoring and cleanup

## [0.2.0] - 2026-01-30

### Added

- CI step for unused dependencies check with `cargo-machete`
- CI step to publish to crates.io
- Default typo extension for 'Tpe'

### Changed

- Remove `serde` dependency (was unused)
- Remove unused `ordered-float` dependency

## [0.1.1] - 2026-01-30

### Fixed

- Minor release fixes

## [0.1.0] - 2026-01-30

### Added

- Initial implementation of the optimization library
- TPE (Tree-structured Parzen Estimator) sampler with optional fixed bandwidth KDE
- Random sampler
- Async optimization support
- CI workflows for testing, coverage (Codecov), auditing, and publishing
- README with project overview, features, and quick start guide

---

## Release Workflow

This changelog is maintained automatically with [git-cliff](https://git-cliff.org/).

1. Write commits using [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, etc.).
2. Tag the release: `git tag v0.X.0`
3. Regenerate the changelog: `make changelog`
4. Commit the updated `CHANGELOG.md` and push.

[0.9.1]: https://github.com/raimannma/rust-optimizer/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/raimannma/rust-optimizer/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/raimannma/rust-optimizer/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/raimannma/rust-optimizer/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/raimannma/rust-optimizer/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/raimannma/rust-optimizer/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/raimannma/rust-optimizer/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/raimannma/rust-optimizer/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/raimannma/rust-optimizer/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/raimannma/rust-optimizer/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/raimannma/rust-optimizer/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/raimannma/rust-optimizer/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/raimannma/rust-optimizer/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/raimannma/rust-optimizer/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/raimannma/rust-optimizer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/raimannma/rust-optimizer/releases/tag/v0.1.0

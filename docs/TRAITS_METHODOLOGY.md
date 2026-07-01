# Traits Methodology

Each driver gets 6 Football-Manager-style trait scores in `[0, 1]`, computed
by `src/traits.py::calculate_driver_traits()` and cached in
`data/cache/driver_traits.json`. Traits are used for the human-readable
explanation text and the Driver Traits dashboard - they are **not** model
inputs (the XGBoost model only sees the numeric features in
`src/features.py`).

## Recency window, not full career

Traits are computed from each driver's **last 2 seasons** of races
(`recent_seasons=2`), not their whole career. If a driver has fewer than
`min_recent_races` (default 5) races in that window - e.g. a rookie, or
someone returning from injury - their full career is used as a fallback
instead of a neutral 0.5 default.

This matters in practice: a driver's rookie season is often full of
retirements and inconsistent results even if they later become a dominant
front-runner. Computing traits over the whole career would permanently drag
down their "consistency" score years after it stopped being true. Using a
recent window means the traits reflect *current form*, matching the mental
model of an FM-style player attribute screen (current ability), not a
career statistics page.

## Why raw standard deviation, not coefficient of variation

`inconsistent` and `tire_management` (its inverse) were originally scored
with the coefficient of variation, `CV = std(position) / mean(position)`.
This has a real flaw: for a driver who wins almost every race (mean
position near 1), even a tiny absolute variance produces a large CV purely
because the denominator is small. In practice this made dominant
front-runners (e.g. a driver winning 19 of 22 races in a season) score as
*more* inconsistent than midfield drivers with genuinely more erratic
results, which is backwards.

The fix (`src/traits.py::_inconsistency_from_std`) normalizes the **raw**
standard deviation of finishing position against a fixed absolute scale
(1-7 positions), calibrated from the observed spread of full-time drivers'
finishing-position std in the dataset:

```python
inconsistency = clip((std_position - 1.0) / (7.0 - 1.0), 0, 1)
```

A driver with std ≤ 1 position is scored as maximally consistent; std ≥ 7 is
scored as maximally inconsistent, linear in between. `tire_management` is
simply `1 - inconsistency`.

## The 6 traits

| Trait | Formula (informal) | Notes |
|---|---|---|
| `qualifying_specialist` | Average (race position − qualifying position), scaled | Positive = gains positions relative to qualifying; negative = loses them |
| `wet_weather_master` | Ratio of average finish on "wet-prone" circuits (Monaco, Silverstone, Spa, Hungaroring heuristic) vs others | No real weather data in the dataset - this is a circuit-based proxy, see [DATA_SOURCES.md](DATA_SOURCES.md) |
| `strong_starter` | Average (grid position − finish position) | Positive = gains positions after the start |
| `tire_management` | `1 - inconsistency` (see above) | |
| `inconsistent` | Normalized std of finishing position (see above) | |
| `track_expert` | How much better a driver's best circuit is vs. their overall average, among circuits raced 3+ times | |

All trait functions return `0.5` (neutral) when there isn't enough data
(fewer than 3-5 relevant races) to compute a meaningful score.

## Known limitation: the wet-weather proxy

There's no weather column in the Kaggle dataset, so `wet_weather_master`
uses a hardcoded list of circuits known for frequent rain as a stand-in. This
is a coarse heuristic - a circuit being "wet-prone" doesn't mean every race
held there was actually wet. Improving this would require joining an
external weather dataset (see `docs/DATA_SOURCES.md` and the Day 8
follow-up in the project roadmap).

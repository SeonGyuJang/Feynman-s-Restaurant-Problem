# Data

## Required Files

| File | Source |
|------|--------|
| `FeynmanStudySamplesObserved.csv` | [OSF](https://osf.io/download/q62ha/) (Christian et al., 2026) |
| `FeynmanStudyData.csv` | [OSF](https://osf.io/download/q62ha/) (optional, for human comparison) |

## File Structure

### FeynmanStudySamplesObserved.csv

| Column | Description |
|--------|-------------|
| `Subject` | Participant ID (0–2519) |
| `Sample Data Observed` | 84 sample scores shown as 3×[28] nested list |
| `Total Nights` | Number of nights (7 / 14 / 28) |
| `Distribution` | Score distribution (uniform / triangular / exponential / power_law) |
| `clamp` | Clamping condition (0–6) |

### LLM Experiment Output (`results/`)

| Column | Description |
|--------|-------------|
| `Subject` | Subject ID (matched to original data) |
| `Total Nights` | Number of nights |
| `Night` | Night index (0-indexed) |
| `Nights Remaining` | Nights left after this decision |
| `Action` | Explore or Exploit |
| `Best Known` | Best score known before this decision |
| `Reward` | Score received this night |
| `Distribution` | Score distribution |
| `clamp` | Clamping condition |
| `Position` | **(New)** Grid coordinate selected by LLM e.g. `(2,3)` |
| `Persona` | **(New)** Persona condition applied |
| `LLM Model` | **(New)** Model used |
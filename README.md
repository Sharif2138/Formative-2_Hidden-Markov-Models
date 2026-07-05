# Human Activity Recognition Using Hidden Markov Models
 
## Overview
 
This project builds a Hidden Markov Model (HMM) to recognize four human activities — standing, walking, jumping, and still — using smartphone accelerometer and gyroscope data. The true activity at any moment can't be observed directly from raw sensor readings, so it is modeled as a hidden state, with sensor-derived features treated as the observable evidence used to infer it.
 
## Repository Contents
 
```
.
├── Formative_2_HMM.ipynb      # Main notebook: data loading through evaluation
├── data/                       # Well-labelled CSV recordings, one folder per activity and more subfolders inside
│   ├── standing_data/
│   ├── walking_data/
│   ├── jumping_data/
│   └── still_data/
|__report.pdf             
└── README.md
```
 
## Data Collection
 
- **App used:** Sensor Logger
- **Device:** [iPhone 11]
- **Sensors recorded:** Accelerometer (x, y, z), Gyroscope (x, y, z)
- **Sampling rate:** ~100.8 Hz (verified using two independent methods — total duration divided by sample count, and average timestamp gap)
- **Recordings:** 50 total, roughly 12–13 per activity, 8-10 seconds each
## Pipeline Summary
 
1. **Loading** — traverses the `data/` folder structure and loads each recording's accelerometer, gyroscope, and metadata into memory.
2. **Quality checks** — verifies no missing values and consistent sampling rate across recordings.
3. **Synchronization** — aligns accelerometer and gyroscope readings per recording using `pandas.merge` since the two sensors only have one row misalignment, which is just an extra row and will be drop by pandas.merge.
4. **Windowing** — splits each synchronized recording into 1-second windows with 50% overlap (~100 samples per window at ~100 Hz).
5. **Feature extraction** — computes 74 features per window: time-domain (mean, variance, std, RMS, SMA, cross-axis correlation) and frequency-domain (dominant frequency, spectral energy, top-5 FFT components), with each signal demeaned before the FFT to avoid the offset dominating the dominant-frequency calculation.
6. **Train/test split** — split by whole recording session (not by individual window) to prevent overlapping windows from leaking between train and test; at least 2 recordings per activity held out for testing.
7. **Scaling** — features standardized with `StandardScaler`, fit only on training data.
8. **HMM training and decoding** — `hmmlearn.GaussianHMM` trained via Baum–Welch, decoded via Viterbi; hidden states mapped to activity labels by majority vote against training labels.
9. **Evaluation** — classification report, confusion matrix, and per-activity sensitivity/specificity/accuracy computed on held-out recordings.
## Experiments
 
**Experiment 1 — Baseline (4 states):** Assumed one hidden state per activity. Converged, but two states both mapped to "jumping," leaving no state for "standing" — resulting in 0% sensitivity for standing and 68.6% overall accuracy.
 
**Diagnostic step:** Boxplots comparing `x_acc_mean` / `y_acc_mean` / `z_acc_mean` between standing and still confirmed the separating signal exists in the raw features, meaning Experiment 1's failure was a model configuration issue, not a data issue.
 
**Experiment 2 — State/seed search:** Searched over `n_states ∈ {4, 5, 6}` and 5 random seeds, selecting the configuration where the most activities were represented by a distinct state. Best configuration: 6 states, seed 0. All four activities were recovered (jumping split naturally across 3 states, reflecting its higher variability), reaching 98% overall test accuracy.
 
## How to Run
 
1. Open `Formative_2_HMM.ipynb` in Google Colab.
2. Mount Google Drive and update `dataset_path` in the loading cell to point to your copy of the `data/` folder.
3. Run all cells top to bottom — Experiment 1 runs first, followed by the diagnostic boxplots, then Experiment 2's state search and final evaluation.
4. Install dependencies if needed:
```bash
   pip install hmmlearn pandas numpy matplotlib seaborn scikit-learn scipy
```
 
## Requirements
 
- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `hmmlearn`
## Key Results
 
| Metric | Experiment 1 (4 states) | Experiment 2 (6 states) |
|---|---|---|
| Overall Accuracy | 0.686 | 0.98 |
| Standing Sensitivity | 0.000 | see `results_df_exp2` in notebook |
| Activities recovered | 3 of 4 | 4 of 4 |
 
Full per-activity metrics, transition matrix heatmaps, confusion matrices, and decoded sequence plots for both experiments are available in the notebook and reproduced in `report.pdf`.
 
## Limitations and Future Work
 
- Single device used for all recordings — sampling rate consistency across multiple devices was not tested.
- Standing vs. still remains the hardest distinction; a dedicated orientation feature (e.g., dominant gravity axis) could strengthen this further.
- Test set is limited to 10 held-out recordings; a larger and more diverse dataset (more sessions, possibly multiple people) would give a more reliable accuracy estimate.
 
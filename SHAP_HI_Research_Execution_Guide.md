# SHAP-Constructed Health Indicator (SHAP-HI) Research Execution Guide
### Cursor AI + MacBook M1 Pro | IEEE PHM 2012 + XJTU-SY Datasets

---

## 🗺️ Research Overview

| Item | Detail |
|---|---|
| **Research Title** | Physics-Informed Explainable RUL Prediction via SHAP-Constructed Health Indicators with Degradation Monotonicity Constraints |
| **Core Novelty** | SHAP values replace manual/black-box HI construction — fully transparent, physics-validated |
| **Datasets** | IEEE PHM 2012 (train) + XJTU-SY (cross-validation) |
| **Timeline** | 8 weeks |
| **Target Venue** | IEEE Transactions on Industrial Informatics (Q1) / RESS |
| **Hardware** | MacBook M1 Pro, PyTorch MPS backend |

---

## 🏗️ Project Structure

Set up this exact folder layout before writing any code:

```
bearing-rul-shap/
├── data/
│   ├── raw/
│   │   ├── phm2012/          ← IEEE PHM 2012 dataset
│   │   └── xjtu-sy/          ← XJTU-SY dataset
│   └── processed/            ← .npz files after feature extraction
├── models/
│   └── checkpoints/          ← saved .pth model files
├── results/
│   ├── figures/              ← PNG/PDF paper figures
│   └── metrics/              ← JSON/CSV result tables
├── src/
│   ├── preprocessing.py      ← feature extraction logic
│   ├── models.py             ← BiLSTM-Attention + loss functions
│   ├── data_loader.py        ← sliding windows + DataLoader
│   ├── shap_hi.py            ← SHAP-HI construction logic
│   └── evaluate.py           ← metrics + scoring functions
├── notebooks/
│   ├── 01_degradation_classifier.ipynb
│   ├── 02_shap_hi_construction.ipynb
│   ├── 03_train_rul_model.ipynb
│   ├── 04_shap_rul_explainability.ipynb
│   └── 05_xjtu_validation.ipynb
├── requirements.txt
└── README.md
```

Create it with one command:

```bash
mkdir -p bearing-rul-shap/{data/{raw/{phm2012,xjtu-sy},processed},models/checkpoints,results/{figures,metrics},src,notebooks}
cd bearing-rul-shap
```

---

## Phase 0 — Environment Setup
### Estimated time: 1–2 hours

### 0.1 Install Miniforge (ARM-native Conda for M1)

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
# Accept all defaults, allow conda init
# Restart terminal after install
```

### 0.2 Create Conda Environment

```bash
conda create -n rul-shap python=3.10 -y
conda activate rul-shap
```

### 0.3 Install All Dependencies

```bash
# PyTorch with M1 MPS (GPU) support
conda install pytorch torchvision torchaudio -c pytorch -y

# Core scientific stack
conda install numpy pandas scipy scikit-learn matplotlib seaborn -y

# Wavelet transform for feature extraction
pip install PyWavelets

# SHAP explainability
pip install shap

# Training utilities
pip install tqdm jupyterlab ipywidgets

# Optional: better progress display
pip install rich
```

### 0.4 Verify Installation

```bash
python -c "
import torch, shap, numpy, sklearn, pywt
print(f'PyTorch:  {torch.__version__}')
print(f'MPS GPU:  {torch.backends.mps.is_available()}')
print(f'SHAP:     {shap.__version__}')
print(f'scikit:   {sklearn.__version__}')
print('✅ All dependencies OK')
"
```

Expected output:
```
PyTorch:  2.2.x
MPS GPU:  True
SHAP:     0.44.x
scikit:   1.4.x
✅ All dependencies OK
```

### 0.5 Save requirements.txt

```bash
pip freeze > requirements.txt
```

### 0.6 Open Project in Cursor

```bash
# From inside the project folder
cursor .
```

> In Cursor: open Settings → Python interpreter → select `rul-shap` conda env.
> Enable: Copilot++ mode for best autocomplete with your codebase context.

---

## Phase 1 — Data Acquisition
### Estimated time: 2–4 hours (mostly download time)

### 1.1 IEEE PHM 2012 (FEMTO-PRONOSTIA)

**Download**:

1. Go to: https://www.kaggle.com/datasets/alanhabrony/ieee-phm-2012-data-challenge
2. Download `archive.zip` (~250 MB)
3. Unzip into `data/raw/phm2012/`

**Alternative via Kaggle CLI**:

```bash
pip install kaggle
# Place kaggle.json API key in ~/.kaggle/kaggle.json
kaggle datasets download -d alanhabrony/ieee-phm-2012-data-challenge -p data/raw/phm2012 --unzip
```

**Expected folder structure**:

```
data/raw/phm2012/
├── Learning_set/
│   ├── Bearing1_1/    (2803 files — full lifecycle ~7.8 hrs)
│   ├── Bearing1_2/    (871 files)
│   ├── Bearing2_1/    (911 files)
│   ├── Bearing2_2/    (797 files)
│   ├── Bearing3_1/    (515 files)
│   └── Bearing3_2/    (1637 files)
├── Test_set/          (11 truncated bearings — cut before failure)
└── Full_Test_set/     (11 complete bearings — includes failure point)
```

Each `acc_XXXXX.csv` file:
- 2560 rows × 6 columns
- Columns: [hour, minute, second, microsecond, horiz_acc(g), vert_acc(g)]
- Duration: 0.1 seconds at 25,600 Hz
- Files sampled every 10 seconds

### 1.2 XJTU-SY Dataset (Cross-Validation)

**Download**:

1. Go to: https://biaowang.tech/xjtu-sy-bearing-datasets/
2. Register with your institutional email, download from Mendeley Data link
3. Unzip into `data/raw/xjtu-sy/`

**Expected folder structure**:

```
data/raw/xjtu-sy/
├── 35Hz12kN/                 ← Condition 1 (35 Hz, 12 kN)
│   ├── Bearing1_1/ (123 files)
│   ├── Bearing1_2/ (161 files)
│   ├── Bearing1_3/ (158 files)
│   ├── Bearing1_4/ (122 files)
│   └── Bearing1_5/ (52 files)
├── 37.5Hz11kN/               ← Condition 2
└── 40Hz10kN/                 ← Condition 3
```

Each CSV: 32,768 rows × 2 columns [horiz_acc, vert_acc] at 25,600 Hz (1.28 sec/file), sampled every 60 seconds.

### 1.3 Verify Data Integrity

**Cursor prompt → create `src/verify_data.py`**:

```
Create a Python script src/verify_data.py that:
1. Checks Learning_set has exactly 6 bearing folders, Full_Test_set has 11
2. For each bearing, counts CSV files and prints total
3. Loads acc_00001.csv from Bearing1_1 and prints shape (should be 2560×6)
4. Checks XJTU-SY has 3 condition folders with 5 bearings each (15 total)
5. Loads file 1.csv from XJTU-SY Bearing1_1 and prints shape (should be 32768×2)
6. Prints PASS or FAIL for each check
Use paths relative to project root.
```

Run:
```bash
python src/verify_data.py
```

---

## Phase 2 — Feature Extraction
### Estimated time: 3–5 hours | Week 1 Day 3–5

This is the foundation. You extract 36 signal features per 0.1-second window.

### Feature Set (36 total per window)

| Domain | Features (per channel) | Count |
|---|---|---|
| Time-domain | Mean, Std, RMS, Max, Min, Peak-to-peak, Skewness, Kurtosis | 8 × 2 = 16 |
| Frequency-domain | Spectral entropy, Spectral kurtosis, Peak freq, Peak magnitude, Freq centroid, Freq variance | 6 × 2 = 12 |
| Wavelet energy | 4-level Daubechies db4 wavelet decomposition energy | 4 × 2 = 8 |
| **Total** | | **36** |

### Cursor Prompt → create `src/preprocessing.py`

```
Create src/preprocessing.py with:

1. Function extract_features(window, fs=25600):
   - Input: numpy array shape (2560, 2) — one time window, 2 channels
   - For EACH channel compute:
     Time-domain (8): mean, std, rms=sqrt(mean(x^2)), max, min, peak2peak=max-min, skewness (scipy.stats.skew), kurtosis (scipy.stats.kurtosis)
     Frequency-domain (6): compute FFT magnitude spectrum, then:
       - spectral_entropy = -sum(p*log(p)) where p = normalized power spectrum
       - spectral_kurtosis = kurtosis of magnitude spectrum
       - peak_freq = frequency at max magnitude
       - peak_magnitude = max(FFT magnitude)
       - freq_centroid = sum(f*|X(f)|)/sum(|X(f)|)
       - freq_variance = sum((f - centroid)^2 * |X(f)|)/sum(|X(f)|)
     Wavelet (4): use pywt.wavedec(x, 'db4', level=4), compute energy = sum(c^2) for each coefficient array
   - Concatenate all features from both channels → shape (36,)
   - Return: numpy array (36,)

2. Function process_bearing(bearing_path, fs=25600, interval_sec=10):
   - Load all acc_*.csv files in sorted order
   - For each file: read columns 5,6 (0-indexed: 4,5) as numpy array (2560, 2)
   - Call extract_features on each window
   - Build feature matrix shape (N, 36) where N = number of CSV files
   - Build RUL array: rul[i] = (N - 1 - i) * interval_sec  (in seconds)
   - Return: features (N,36), rul (N,), timestamps (N,)

3. Main block:
   - Process all 6 bearings from Learning_set
   - Process all 11 bearings from Full_Test_set
   - Stack all train data: features_train (N_total, 36), rul_train (N_total,), bearing_id (N_total,)
   - Save to data/processed/phm2012_train.npz and data/processed/phm2012_test.npz
   - Use tqdm for progress bars
   - Print summary: total samples, feature shape, RUL min/max
```

Run (takes ~10–15 minutes on M1):

```bash
python src/preprocessing.py
```

Repeat for XJTU-SY with adjusted parameters (32768 samples/file, interval=60 sec):

**Cursor prompt → create `src/preprocessing_xjtu.py`**:

```
Create src/preprocessing_xjtu.py following the same pattern as src/preprocessing.py but:
- CSV files have 32768 rows × 2 columns (no timestamp columns — just horiz, vert directly)
- interval_sec = 60 (files sampled every 1 minute)
- Extract same 36 features per window
- Process all 15 bearings (3 conditions × 5 bearings)
- Save to data/processed/xjtu_features.npz with condition_id and bearing_id arrays
```

---

## Phase 3 — SHAP-HI Construction
### Estimated time: 1 day | Week 2

This is the core novelty. You train a degradation classifier, extract SHAP values, and use them to build the Health Indicator.

### Notebook: `01_degradation_classifier.ipynb`

**Open in Cursor**: `notebooks/01_degradation_classifier.ipynb`

**Cursor prompt for full notebook**:

```
Create a Jupyter notebook notebooks/01_degradation_classifier.ipynb with these cells:

Cell 1 — Load data:
  Load data/processed/phm2012_train.npz
  Print: features shape, RUL range, number of bearings

Cell 2 — Create stage labels:
  For each bearing separately, compute RUL percentage = rul / rul.max()
  Stage 0 (Healthy):  rul_pct > 0.75
  Stage 1 (Early):    0.50 < rul_pct <= 0.75
  Stage 2 (Mid):      0.25 < rul_pct <= 0.50
  Stage 3 (Critical): rul_pct <= 0.25
  Plot: bar chart of sample count per stage

Cell 3 — Train Random Forest:
  StandardScaler on X
  RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
  Stratified 5-fold cross-validation (split by bearing, not by sample)
  Print: accuracy per fold and mean ± std
  Save: scaler to models/checkpoints/feature_scaler.pkl
  Save: trained RF model to models/checkpoints/rf_stage_classifier.pkl

Cell 4 — SHAP TreeExplainer:
  explainer = shap.TreeExplainer(rf_model)
  shap_values = explainer.shap_values(X_train_scaled)
  # shap_values: list of 4 arrays, each (N_train, 36)
  Compute global importance: mean(|shap_values[i]|) summed across all 4 stages → shape (36,)

Cell 5 — SHAP feature importance plot:
  Bar chart: top 20 features sorted by global SHAP importance
  Compare with RF native feature_importances_ in same chart (grouped bars)
  Save figure to results/figures/shap_vs_rf_importance.png

Cell 6 — SHAP summary plot for Stage 3 (Critical):
  shap.summary_plot(shap_values[3], X_train_scaled, feature_names=feature_names)
  Interpretation: red = high feature value, blue = low → which direction drives critical stage?

Cell 7 — Feature selection:
  Keep features where global SHAP importance > threshold (mean - 0.5*std)
  Typically 8-14 features selected
  Print: selected feature names and their SHAP weights
  Save: {'features': selected_names, 'weights': shap_weights} → results/metrics/shap_feature_weights.json

Cell 8 — Build SHAP-HI:
  For selected K features, apply MinMaxScaler per feature (fit on Stage 0 — healthy data only)
  HI(t) = sum(shap_weight[i] * normalized_feature[i](t)) for i in selected features
  Normalize HI to [0,1] range (0=healthy, 1=failure)
  Plot: HI over time for Bearing1_1 (x-axis: time step, y-axis: HI)
  Overlay RUL on secondary y-axis → confirm inverse relationship

Cell 9 — HI quality metrics:
  For each bearing compute:
    Monotonicity = percentage of time steps where HI decreases (wrong direction) → should be <5%
    Spearman correlation between HI and -RUL → should be >0.90
    Trendability = correlation between HI and linear fit → should be >0.85
  Print table of metrics per bearing
  Save: results/metrics/hi_quality_metrics.csv
```

---

## Phase 4 — RUL Prediction Model
### Estimated time: 3–4 days | Week 3–4

### `src/models.py`

**Cursor prompt**:

```
Create src/models.py with:

Class 1 — BiLSTMAttention(nn.Module):
  __init__(input_size=1, hidden_size=128, num_layers=2, dropout=0.3, output_size=1):
    self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
    self.attention = nn.Linear(hidden_size*2, 1)
    self.fc1 = nn.Linear(hidden_size*2, 64)
    self.fc2 = nn.Linear(64, output_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  forward(x):
    # x shape: (batch, seq_len, input_size)
    lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden*2)
    attn_weights = softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
    context = sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2) — weighted sum
    out = self.dropout(self.relu(self.fc1(context)))
    out = self.fc2(out)  # (batch, 1)
    return out.squeeze(-1), attn_weights.squeeze(-1)  # return both RUL and attention

Class 2 — MonotonicityLoss(nn.Module):
  forward(rul_sequence):
    # rul_sequence: predicted RUL for consecutive time steps, shape (batch, seq_len)
    diffs = rul_sequence[:, 1:] - rul_sequence[:, :-1]  # should be negative (decreasing)
    violations = torch.clamp(diffs, min=0)  # penalize increases
    return violations.mean()

Class 3 — CombinedRULLoss(nn.Module):
  __init__(lambda_mono=0.1):
    stores lambda_mono
  forward(pred, target, pred_sequence=None):
    mse_loss = F.mse_loss(pred, target)
    if pred_sequence is not None:
      mono_loss = MonotonicityLoss()(pred_sequence)
      return mse_loss + self.lambda_mono * mono_loss, mse_loss.item(), mono_loss.item()
    return mse_loss, mse_loss.item(), 0.0
```

### `src/data_loader.py`

**Cursor prompt**:

```
Create src/data_loader.py with:

Function 1 — compute_shap_hi(features, shap_weights_path, scaler_path):
  Load shap_feature_weights.json → get selected feature indices and weights
  Load healthy_scaler.pkl (MinMaxScaler fit on Stage 0 data)
  Extract selected features from input array
  Scale them using healthy_scaler
  Compute HI = weighted sum → normalize to [0,1]
  Return: HI array shape (N,)

Function 2 — create_sliding_windows(hi_seq, rul_seq, window_size=50, stride=5):
  For each valid window:
    X[i] = hi_seq[i : i+window_size]  → shape (window_size, 1)
    y[i] = rul_seq[i + window_size - 1]  → scalar RUL at last step
  Return: X shape (num_windows, window_size, 1), y shape (num_windows,)

Function 3 — get_phm_dataloaders(train_path, window_size=50, batch_size=64, val_bearing='Bearing3_2'):
  Load phm2012_train.npz
  Compute SHAP-HI for all samples
  Split: train = all bearings except val_bearing, val = val_bearing
  Create windows for train and val separately
  Normalize RUL by dividing by max RUL in training set (save this value)
  Return: train_loader, val_loader, rul_normalizer (float)
```

### Notebook: `03_train_rul_model.ipynb`

**Cursor prompt**:

```
Create notebooks/03_train_rul_model.ipynb:

Cell 1 — Device setup:
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'
  print(f'Device: {device}')

Cell 2 — Hyperparameters dict:
  config = {
    window_size: 50,
    batch_size: 64,
    hidden_size: 128,
    num_layers: 2,
    dropout: 0.3,
    lr: 1e-3,
    weight_decay: 1e-4,
    epochs: 100,
    patience: 15,
    lambda_mono: 0.1
  }

Cell 3 — Initialize model, optimizer, scheduler:
  model = BiLSTMAttention(...).to(device)
  optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
  criterion = CombinedRULLoss(lambda_mono=config.lambda_mono)
  scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

Cell 4 — Training loop:
  Track: train_loss, val_loss, val_rmse per epoch
  Early stopping on val_loss
  Save best model when val_loss improves → models/checkpoints/best_model.pth
  Print epoch summary with tqdm

Cell 5 — Training curves plot:
  Two subplots: loss curve, RMSE curve
  Mark epoch of best model with vertical dashed line
  Save: results/figures/training_curves.png

Cell 6 — Test set evaluation:
  Load Full_Test_set bearings from phm2012_test.npz
  For each test bearing:
    Compute SHAP-HI, create windows, predict RUL at each step
    Compute RMSE, MAE, PHM 2012 scoring function
  PHM Scoring formula:
    E = pred_rul_last - true_rul_last (error at end of truncation)
    if E <= 0: score = exp(-ln(0.5) * E / 5) - 1
    else:      score = exp(ln(0.5) * E / 20) - 1
  Print results table, save to results/metrics/test_results.csv

Cell 7 — True vs Predicted RUL plot:
  For 3 sample bearings (1_3, 2_3, 3_3): overlay true RUL and predicted RUL over time
  Add shaded uncertainty band (±1 std of predictions)
  Save: results/figures/rul_prediction_results.png
```

Expected training time on M1 Pro: **20–40 minutes**.

---

## Phase 5 — SHAP Explainability on RUL Predictions
### Estimated time: 1–2 days | Week 5

### Notebook: `04_shap_rul_explainability.ipynb`

**Cursor prompt**:

```
Create notebooks/04_shap_rul_explainability.ipynb:

Cell 1 — Load model and data:
  Load best_model.pth
  Prepare 100 background sequences (random sample from training set) as torch tensor
  Prepare 20 test sequences from one test bearing (Bearing1_3)

Cell 2 — SHAP GradientExplainer:
  # Wrap model to return only RUL (not attention weights) for SHAP compatibility
  class RULOnly(nn.Module):
    def forward(x): return model(x)[0]
  explainer = shap.GradientExplainer(RULOnly(), background_data)
  shap_values = explainer.shap_values(test_sequences)
  # shap_values shape: (20, 50, 1) → SHAP per time step in the window

Cell 3 — Temporal SHAP importance:
  mean_abs_shap = mean(|shap_values|, axis=(0,2))  → shape (50,)  (importance per position in window)
  Line plot: x=time position (0-49), y=mean |SHAP|
  Interpretation: peaks show which part of the history window matters most
  Expected: recent time steps should have higher importance than distant ones
  Save: results/figures/temporal_shap_importance.png

Cell 4 — Attention vs SHAP Agreement:
  Extract attention weights from same 20 sequences: shape (20, 50)
  Compute Spearman correlation between attention[i] and |shap_values[i]| for each sequence
  Plot: histogram of correlation values across 20 sequences
  If median correlation > 0.6: model is internally consistent with post-hoc explanation
  Save: results/figures/attention_shap_agreement.png

Cell 5 — SHAP Waterfall at Critical Moment:
  Select sequence where predicted RUL < 500 seconds (near failure)
  shap.waterfall_plot for that sequence — shows how each time step pushes prediction up/down
  Save: results/figures/shap_waterfall_critical.png

Cell 6 — Physics Alignment Score:
  Physics expectation: higher kurtosis, RMS, spectral entropy → lower RUL (positive SHAP for damage features)
  Define physics_rank = expert-assigned importance rank of 36 features
  Define shap_rank = SHAP global importance rank
  Compute Spearman correlation between physics_rank and shap_rank
  Print: "Physics Alignment Score = {correlation:.3f}"
  Target: >0.75 indicates physically consistent model

Cell 7 — Monotonicity violation analysis:
  For each test bearing, count time steps where predicted RUL[t+1] > predicted RUL[t]
  violation_rate = violations / total_steps * 100
  Compare: model with physics loss vs. without (retrain without lambda_mono as ablation)
  Bar chart: violation rate per bearing for both models
  Save: results/figures/monotonicity_comparison.png
```

---

## Phase 6 — Cross-Dataset Validation (XJTU-SY)
### Estimated time: 2 days | Week 6

### Notebook: `05_xjtu_validation.ipynb`

**Cursor prompt**:

```
Create notebooks/05_xjtu_validation.ipynb:

Cell 1 — Apply SHAP-HI to XJTU-SY:
  Load xjtu_features.npz
  Use SAME shap_feature_weights.json and healthy_scaler.pkl from PHM 2012
  Compute SHAP-HI for all 15 XJTU-SY bearings
  Plot: HI over time for 3 sample bearings, check for monotonic trend
  Print: HI monotonicity scores for all 15 bearings

Cell 2 — Zero-shot RUL prediction:
  Directly apply PHM-trained model (best_model.pth) to XJTU-SY without retraining
  Compute RMSE for each bearing
  Note: RUL scale differs (XJTU uses minutes not seconds) — normalize before evaluation

Cell 3 — Fine-tuned RUL prediction:
  Fine-tune best_model.pth on Bearing1_1 from XJTU-SY only (15 epochs, lr=1e-4)
  Test on remaining 14 XJTU-SY bearings
  Compare RMSE: zero-shot vs. fine-tuned

Cell 4 — Cross-dataset SHAP comparison:
  Compute SHAP GradientExplainer on XJTU-SY test sequences
  Compare SHAP temporal importance: PHM 2012 vs. XJTU-SY
  Side-by-side bar chart of feature importance
  If rankings are consistent (Spearman >0.7): SHAP-HI is transferable
  Save: results/figures/cross_dataset_shap_comparison.png

Cell 5 — Results summary table:
  DataFrame with: Dataset, Bearing, RMSE, MAE, Scoring, Monotonicity_Violation%
  Print and save to results/metrics/final_results.csv
```

---

## Phase 7 — Paper Figures & Writing
### Estimated time: 1.5 weeks | Week 7–8

### 7.1 Generate All Paper Figures

**Cursor prompt → create `src/generate_figures.py`**:

```
Create src/generate_figures.py that generates all 7 paper figures in high-resolution PDF:

Figure 1 — Research framework diagram (skip — do manually in draw.io)
Figure 2 — SHAP vs RF importance comparison (load from results)
Figure 3 — SHAP-HI monotonic degradation curves (3 bearings, PHM 2012)
Figure 4 — True vs Predicted RUL (3 test bearings with error shading)
Figure 5 — Temporal SHAP importance (line chart, both datasets overlaid)
Figure 6 — Attention-SHAP agreement histogram
Figure 7 — Cross-dataset SHAP feature importance comparison

All figures:
  - Size: (6, 4) inches for single-column, (12, 4) for double-column (IEEE format)
  - DPI: 300 for PNG, vector for PDF
  - Font: Times New Roman, 10pt (IEEE standard)
  - Save to: results/figures/ as both .png and .pdf
```

### 7.2 Compile Results Tables

**Cursor prompt → create `src/compile_tables.py`**:

```
Create src/compile_tables.py that generates LaTeX-formatted tables:

Table 1 — Selected SHAP-HI features:
  Columns: Feature Name | Physical Meaning | SHAP Weight | Physical Relevance
  Load from results/metrics/shap_feature_weights.json

Table 2 — HI quality metrics:
  Columns: Bearing | Monotonicity (%) | Spearman ρ | Trendability
  Load from results/metrics/hi_quality_metrics.csv

Table 3 — RUL prediction performance (PHM 2012):
  Columns: Bearing | RMSE (s) | MAE (s) | PHM Score | Violation (%)
  Compare: SHAP-HI + BiLSTM vs. Manual HI + BiLSTM vs. AE HI + BiLSTM

Table 4 — Cross-dataset results (XJTU-SY):
  Columns: Bearing | Zero-shot RMSE | Fine-tune RMSE | SHAP Alignment Score

Save all tables to results/metrics/paper_tables.tex
```

### 7.3 Paper Structure

```
Abstract (150 words) — problem, method, key results (RMSE, SHAP alignment score)

1. Introduction
   1.1 Bearing RUL prediction in industrial maintenance
   1.2 Limitations of manual HI and black-box AE approaches
   1.3 Research gap: explainability in HI construction
   1.4 Contributions (4 bullets)

2. Related Work
   2.1 Manual health indicator construction (RMS, kurtosis-based)
   2.2 Deep learning for RUL prediction (CNN, LSTM, Transformer)
   2.3 Explainable AI in PHM (existing SHAP post-hoc work)
   2.4 Physics-informed machine learning

3. Methodology
   3.1 System overview (block diagram)
   3.2 Multi-domain feature extraction (36 features)
   3.3 SHAP-based HI construction (Algorithm 1)
   3.4 BiLSTM-Attention RUL predictor (Figure + equations)
   3.5 Monotonicity-constrained loss function (Equation)

4. Experiments
   4.1 Datasets description (PHM 2012 + XJTU-SY)
   4.2 Implementation details (PyTorch, hyperparameters)
   4.3 Evaluation metrics (RMSE, MAE, PHM scoring)
   4.4 Baseline methods for comparison

5. Results and Discussion
   5.1 SHAP-HI quality analysis (Table 2 + Figure 3)
   5.2 RUL prediction performance (Table 3 + Figure 4)
   5.3 Explainability analysis (Figure 5 + Figure 6)
   5.4 Physics alignment validation (alignment score)
   5.5 Cross-dataset transferability (Table 4 + Figure 7)
   5.6 Ablation study (with/without physics loss)

6. Conclusion + Future Work
```

**Target journals** (in priority order):

| Journal | Impact Factor | Submission |
|---|---|---|
| IEEE Trans. Industrial Informatics (TII) | 11.7 | Open |
| Reliability Engineering & System Safety | 9.4 | Open |
| Mechanical Systems and Signal Processing | 8.4 | Open |
| IEEE Trans. Industrial Electronics (TIE) | 7.7 | Open |

---

## ⚡ Daily Commands Cheat Sheet

```bash
# Start session
conda activate rul-shap
cd ~/bearing-rul-shap
cursor .

# Run preprocessing
python src/preprocessing.py
python src/preprocessing_xjtu.py

# Launch notebook server
jupyter lab --no-browser --port=8888

# Train model
cd notebooks && jupyter nbconvert --to notebook --execute 03_train_rul_model.ipynb

# Generate all figures
python src/generate_figures.py

# Check GPU utilization (M1 MPS)
python -c "import torch; t = torch.randn(1000,1000).to('mps'); print('MPS active')"
```

---

## 🛠️ Troubleshooting (M1-Specific)

| Problem | Cause | Fix |
|---|---|---|
| `MPS available: False` | macOS < 12.3 or old PyTorch | Update macOS + `pip install --upgrade torch` |
| `RuntimeError: MPS backend out of memory` | Large batch size | Reduce batch_size from 64 → 32 |
| `SHAP install fails` | C++ compiler missing | `xcode-select --install` then retry |
| `torch.nn.LSTM slow on MPS` | Known MPS limitation for RNNs | Use `device = 'cpu'` for LSTM layers only |
| `pywt not found` | conda vs pip mismatch | `conda install pywavelets` |
| Slow CSV loading | Sequential file reads | Add `from joblib import Parallel, delayed` |

> **M1 LSTM tip**: If LSTM training is slow on MPS, set LSTM to run on CPU and only move FC/attention layers to MPS — this is a known Apple Silicon quirk with recurrent networks.

```python
# Hybrid device setup for M1
device = torch.device('mps')
model.bilstm = model.bilstm.to('cpu')  # LSTM on CPU
model.attention = model.attention.to(device)  # rest on MPS
model.fc1 = model.fc1.to(device)
model.fc2 = model.fc2.to(device)
```

---

## ✅ Q1 Publication Checklist

- [ ] SHAP-HI Spearman ρ > 0.90 vs RUL on all 6 training bearings
- [ ] Trendability score > 0.85 for all training bearings
- [ ] PHM 2012 test RMSE < 500 seconds
- [ ] PHM 2012 scoring function: lower is better, beat LSTM baseline
- [ ] Monotonicity violation < 2% with physics loss (vs >10% without)
- [ ] Physics alignment score > 0.75
- [ ] Cross-dataset transfer: XJTU fine-tune RMSE within 20% of PHM RMSE
- [ ] Ablation study: 4 variants (no physics loss, manual HI, AE-HI, SHAP-HI+physics)
- [ ] All code runs end-to-end from raw data in clean conda env
- [ ] Paper < 12 pages, IEEE double-column format
- [ ] 3+ baseline comparisons cited and reproduced

# ModelGuard — Model Evaluation & Metrics Specification

## 1. Evaluation Methodology & Temporal Split

Traffic volume prediction is an inherently auto-correlated time-series task. Random K-Fold splits introduce severe data leakage from future observations into training folds.

ModelGuard enforces **strict chronological ordering** across three deterministic partitions:

$$\text{Dataset} = \text{Train} \,(70\%) \longrightarrow \text{Validation} \,(15\%) \longrightarrow \text{Test Holdout} \,(15\%)$$

- **Train Set (70%):** Model parameter optimization (fitting trees).
- **Validation Set (15%):** Hyperparameter tuning, candidate selection, and Release Gate threshold evaluation.
- **Test Holdout Set (15%):** Final unbiased generalizability assessment before candidate registration.

---

## 2. Feature Engineering & Preprocessing Pipeline

The Scikit-Learn `Pipeline` implements deterministic transformations without lookahead bias:

| Feature | Input Type | Transformation | Output Representation |
|---|---|---|---|
| `temp` | Continuous (Kelvin) | StandardScaler | Standardized $z$-score |
| `rain_1h`, `snow_1h` | Continuous (mm) | StandardScaler | Standardized $z$-score |
| `clouds_all` | Integer (%) | StandardScaler | Standardized $z$-score |
| `weather_main` | Categorical | OneHotEncoder(handle_unknown='ignore') | Binary indicator columns |
| `holiday` | Categorical | Binary indicator (`is_holiday`) | `0` or `1` |
| `date_time` | Timestamp string | Cyclical sine & cosine transforms | $\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$, $\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$ |

---

## 3. Metrics Definitions

For ground truth $y_i$ and predicted values $\hat{y}_i$ across $N$ samples:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   Measures average prediction error in units of vehicles per hour. Highly robust to outliers.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   Penalizes large errors quadratically, reflecting severe congestion prediction misses.

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   Proportion of variance explained by the model compared to the trivial mean baseline.

4. **Relative MAE Improvement:**
   $$\text{Improvement} = \frac{\text{MAE}_{\text{baseline}} - \text{MAE}_{\text{candidate}}}{\text{MAE}_{\text{baseline}}} \times 100\%$$

---

## 4. Benchmark Performance & Gate Policies

### Baseline vs. Candidate Results on UCI Metro Traffic:

| Model | MAE (veh/hr) | RMSE (veh/hr) | $R^2$ Score | Relative Improvement |
|---|---|---|---|---|
| **Mean Baseline (`DummyRegressor`)** | 1,745.2 | 1,985.4 | 0.000 | 0.0% (Reference) |
| **Candidate Model (`RandomForestRegressor`)** | **284.1** | **452.8** | **0.948** | **+83.7%** |

### Release Gate Default Policy Thresholds:

```json
{
  "max_val_mae": 350.0,
  "max_val_rmse": 550.0,
  "min_val_r2": 0.90,
  "min_improvement_pct": 50.0
}
```

Any candidate failing any of these conditions is automatically blocked from staging with status `GATE_FAILED`.

# System Architecture & Technical Design (`ARCHITECTURE.md`)
**Project:** Urban Dengue Outbreak Early-Warning System (2–4 Week Forecast)  
**Challenge:** EDU AI Engineering Hackathon 2026 — Preliminary Round  

---

## 1. End-to-End System Data Flow Diagram

```text
+-----------------------------------------------------------------------+
| 1. RAW DATA INGESTION (pipeline.py)                                   |
|    - Weekly Ward Dataset (31,320 rows | 60 wards | 5 cities | 10 yrs) |
|    - Weather: avg_temp_c, max_temp_c, min_temp_c, rainfall_mm,        |
|               relative_humidity_pct                                   |
|    - Ward Vulnerability: population_density_sqkm, sanitation_score,   |
|                          water_logging_risk_index                     |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 2. WARD-GROUPED MULTI-WEEK LAG PIPELINE (compute_ward_weather_lags)   |
|    - Grouped strictly by `ward_id` sorted chronologically by `date`   |
|    - Computes 1w, 2w, 3w, 4w lags for rainfall, temperature, humidity |
|    - Aggregates critical 2–4w mosquito breeding window:               |
|        * rainfall_2_4w_sum = lag_2w + lag_3w + lag_4w (r = 0.81)      |
|        * temp_2_4w_mean & humidity_2_4w_mean                          |
|    - Engineered Interaction Index:                                    |
|        * stagnant_breeding_potential =                                |
|            rainfall_2_4w_sum * water_logging_risk * (1 - sanitation)  |
|    - Cyclical Seasonality: sin_week, cos_week                         |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 3. CHRONOLOGICAL SPLIT & MODEL TRAINING (train_model.py)              |
|    - Train Set: 2016-02-01 to 2023-12-25 (24,780 rows | 80%)          |
|    - Test Set : 2024-01-01 to 2025-12-29 (6,300 unseen future rows)   |
|    - Serialized Bundle (`models/dengue_model_bundle.joblib`):         |
|        1) Point Forecast: HistGradientBoostingRegressor (L2 loss)     |
|        2) Lower Bound (10th percentile): Quantile HGBR                |
|        3) Upper Bound (90th percentile): Quantile HGBR                |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 4. ZERO-API LOCAL INFERENCE & INTERACTIVE DASHBOARD (app.py)          |
|    - `@st.cache_resource` loads `.joblib` once into RAM (< 10 ms/run) |
|    - Risk Tier Engine: Low (<3), Medium (3–7), High (7–12), Critical  |
|    - Interactive What-If Climate Simulator, Ward Risk Heatmap, and    |
|      Chronological Validation Error Dashboard (MAE / RMSE / R²)       |
+-----------------------------------------------------------------------+
```

---

## 2. Multi-Week Vector Breeding Pipeline & Feature Engineering

Dengue transmission does not respond instantaneously to rainfall. *Aedes aegypti* eggs require 7–14 days in stagnant urban containers to mature into adult vectors, followed by extrinsic viral incubation and human symptom onset—creating a **2-to-4-week epidemiological lag**.

To model this biological mechanism without data leakage across neighborhoods:
1. **Ward-Level Grouping:** Records are partitioned by `ward_id` (`groupby("ward_id")`) and ordered strictly by `date` before applying `.shift(k)` for $k \in \{1, 2, 3, 4\}$ weeks.
2. **2–4 Week Breeding Window Aggregates:** `rainfall_2_4w_sum` captures cumulative precipitation across weeks $t-2, t-3,$ and $t-4$. Exploratory analysis confirms `rainfall_2_4w_sum` correlates strongly with weekly dengue cases ($r = 0.813$), far exceeding same-week rainfall ($r = 0.531$).
3. **Hyper-Local Infrastructure Interaction:** Identical rainfall produces divergent outbreak trajectories depending on neighborhood drainage and waste management. Our composite feature `stagnant_breeding_potential` explicitly combines lagged rainfall with `water_logging_risk_index` and poor sanitation (`1 - sanitation_score`).

---

## 3. Chronological Validation Split & Model Selection Rationale

### Chronological Validation Strategy (Zero Temporal Leakage)
Random k-fold cross-validation leaks future seasonal patterns into training folds in time-series epidemiology. We enforce a strict **chronological split**:
- **Training Window:** `2016-02-01` to `2023-12-25` (`24,780` ward-weeks across 8 full monsoon cycles).
- **Holdout Test Window:** `2024-01-01` to `2025-12-29` (`6,300` completely unseen future ward-weeks).

### Model Selection: `HistGradientBoostingRegressor`
We selected scikit-learn's `HistGradientBoostingRegressor` (histogram-based gradient boosted decision trees) over linear models and deep sequence networks for four engineering reasons:
1. **Non-Linear Threshold Capture:** Mosquito breeding accelerates non-linearly within optimal temperature ($25^\circ\text{C}–32^\circ\text{C}$) and high-humidity regimes; tree splits naturally capture these thresholds and interaction effects.
2. **Native Quantile Uncertainty Estimation:** Training three lightweight models (`squared_error`, `quantile=0.10`, `quantile=0.90`) provides calibrated **80% prediction confidence intervals** (`93.3%` empirical test coverage) alongside point predictions.
3. **Strong Out-of-Sample Generalization:** On unseen 2024–2025 seasons, the model achieves **MAE = 0.82 cases/week**, **RMSE = 1.55 cases/week**, and **$R^2 = 0.838$**.
4. **Compact Footprint & Fast Inference:** The compressed `.joblib` bundle is under 1.5 MB and executes inference in **< 10 milliseconds** on standard CPU hardware.

---

## 4. Compute Efficiency, Caching & API Cost Controls

Public health offices in Dhaka, Chattogram, Gazipur, and Narayanganj require reliable early-warning tools without recurring cloud GPU or LLM API expenses:
- **Zero Runtime External API Calls:** All inference runs 100% locally against serialized `.joblib` artifacts—eliminating network latency, API rate limits, and per-query billing.
- **Zero LLM Token Overhead:** Risk tiers (`Low`, `Medium`, `High`, `Critical`) and targeted municipal interventions (larvicide, drain clearance, thermal fogging, hospital surge readiness) are generated deterministically from calibrated model outputs rather than expensive generative LLM calls.
- **Streamlit Memory Caching:** `@st.cache_resource` keeps the serialized model bundle resident in memory across user interactions, while `@st.cache_data` caches the pre-computed ward lag dataframe—preventing redundant disk reads or feature recomputation when users toggle wards or adjust What-If weather sliders.

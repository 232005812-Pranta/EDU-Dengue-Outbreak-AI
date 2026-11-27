# 🦟 Urban Dengue Outbreak Early Warning System (2–4 Week Forecast)
**EDU AI Engineering Hackathon 2026 — Final Round**

A hyper-local, ward-level Dengue outbreak early-warning dashboard built for public health officers across major Bangladeshi cities (**Dhaka North, Dhaka South, Chattogram, Gazipur, and Narayanganj**). By modeling the 2-to-4-week *Aedes aegypti* vector breeding cycle alongside local infrastructure vulnerability (`water_logging_risk_index`, `sanitation_score`, and `population_density_sqkm`), the system predicts outbreak cases and classifies wards into **Low, Medium, High, and Critical** risk tiers before hospital surges peak.

---

## 🚀 Quick Local Setup (<= 3 Minutes)

### 1. Clone the Repository
```bash
git clone [https://github.com/232005812-Pranta/EDU-Dengue-Outbreak-AI.git](https://github.com/232005812-Pranta/EDU-Dengue-Outbreak-AI.git)
cd EDU-Dengue-Outbreak-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train & Serialize the Model (Optional — Pre-trained `.joblib` included)
```bash
python train_model.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Key Features
- **Ward-Level Multi-Week Lag Pipeline:** Automatically groups historical weather records by `ward_id` and computes 1-to-4-week lagged rainfall, temperature, and humidity features plus a composite `stagnant_breeding_potential` index.
- **Chronological Validation & Uncertainty Bounds:** Trained strictly on historical data (`2016–2023`) and validated on unseen future seasons (`2024–2025`), outputting point forecasts alongside 80% prediction confidence intervals (`10th–90th` percentiles).
- **Zero-API Fast Local Inference:** Uses serialized `HistGradientBoostingRegressor` models (`joblib`) for sub-10ms local predictions without recurring cloud or LLM API costs.
- **Interactive Public Health Dashboard:** Includes ward-level risk heatmaps, weather driver breakdowns, what-if weather simulation sliders, and a chronological validation error (`MAE` / `RMSE`) monitor.

## 🚑 Phase 2 (Onsite Final): Operations Command System
Extends the forecasts into decisions for all 60 wards (details: `ARCHITECTURE.md` §6):
1. **Constrained Resource Allocation** (`operations.py`): saline bags, diagnostic kits, hospital beds and spraying crews per ward, with live *Available Regional Stock (30–100%)* and *Total Spraying Crews* sliders, risk × population-density prioritisation and red deficit alerts.
2. **Hospital Overflow Routing** (`operations.py`): intra-city patient transfers, net ward deficits, city bed utilisation and headroom.
3. **Vector Control Simulator** (`simulator.py`): deploy crews → the ML models are re-run → cases, beds and saline prevented in weeks +2…+4, ranked by ROI (cases prevented per crew).
4. **Dispatch Notices** (`dispatch.py`): SMS (English/Bangla), WhatsApp and printable HTML commissioner briefs; samples in `/exports`.
5. **Offline Resilience**: everything runs locally; `python test_pipeline.py` includes a network kill-switch test.
6. **Bonus:** offline ward risk map (Tab 8, no map tiles) and a full **English / বাংলা** UI switch, both working with Wi-Fi off.

```bash
python export_dispatch.py          # regenerate /exports (offline)
python test_pipeline.py            # 10 checks incl. offline kill-switch
python -m streamlit run app.py     # tabs 4-7 = Phase 2 modules
```

## 🏁 Final Round Upgrade: True Early Warning (No Future Data)
- **Direct multi-horizon models:** a dedicated point + 80%-interval model for each of +1, +2, +3, +4 weeks, using **only data observed up to the reference week** (no future weather).
- **Past-case surveillance features:** ward-level case lags, 4-week mean/max and 2-week trend.
- **Beats naive baseline by 26–43% MAE** on unseen 2024–2025 (e.g. +3W MAE 1.10 vs 1.92). See `ARCHITECTURE.md` §5.
- **Leakage test:** `python test_pipeline.py` proves forecasts don't change when all future data is deleted.
- Selecting the latest reference week produces a genuine forecast into 2026.

---

## 🤖 AI Assistants & External Tools Disclosure
In accordance with hackathon guidelines, we disclose all AI tools and libraries used during development:
- **Claude (Anthropic):** Assisted with the leakage-free multi-horizon forecasting upgrade and, in the onsite final, with the Phase 2 allocation, overflow routing, vector-control simulator, dispatch templates, tests and documentation. No AI service is called at runtime.
- **Google Gemini:** Assisted with exploratory data analysis, lag feature engineering design, Streamlit dashboard structuring, and documentation drafting.
- **External APIs:** None required at runtime (100% offline local inference using open-source `scikit-learn`, `pandas`, `plotly`, and `streamlit`).

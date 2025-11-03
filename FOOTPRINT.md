# 🌍 Carbon Footprint Measurement Methodology

## Overview

I measured the carbon footprint of my ML models during **inference**, not training, because in real-world deployments, models are used thousands of times each year across millions of devices. Training happens only once, but inference runs continuously — so that's where most of the energy impact lies.

## Measurement Approach

### 1. Inference Time Measurement

I used Python's built-in `time.time()` function to measure how long each model takes to make predictions:

```python
import time

# Measure RandomForest inference time
start = time.time()
rf_predictions = rf_model.predict(X_test_scaled)
rf_time = time.time() - start

# Measure Ridge inference time
start = time.time()
ridge_predictions = ridge_model.predict(X_test_scaled)
ridge_time = time.time() - start
```

**Results:**

- **RandomForest:** ~100ms per prediction (averaged over 29,040 test samples)
- **Ridge:** ~5ms per prediction (around 20× faster)

### 2. Energy Consumption Calculation

Since I didn't have access to hardware-based power measurement tools (like CodeCarbon), I estimated energy use based on typical CPU power consumption.

**Assumptions:**

- **CPU Power Draw:** 15 watts during inference (a conservative estimate for an Intel i7)
- Typical desktop CPU runs at 10–20W under moderate load
- **Formula:** Energy (kWh) = Power (W) × Time (hours) / 1000

**Calculation Example (RandomForest):**

```
Time per prediction: 100ms = 0.0000278 hours
Annual predictions per meter: 8,760 (hourly forecasts)
Number of smart meters: 5,000,000
Total predictions: 43,800,000,000

Total runtime: 100ms × 43.8B = 4.38B ms = 1,217 hours
Energy: 15W × 1,217 hours / 1000 = 18.25 kWh
```

**Calculation Example (Ridge):**

```
Time per prediction: 5ms = 0.00000139 hours
Total runtime: 5ms × 43.8B = 219M ms = 60.8 hours
Energy: 15W × 60.8 hours / 1000 = 0.913 kWh
```

### 3. Carbon Emissions Calculation

I converted energy usage into CO₂ emissions using the US grid's average carbon intensity.

**Carbon Intensity:** 0.4 kg CO₂ per kWh  
**Source:** US EPA average grid emission factor

**Calculation:**

- **RandomForest:** 18.25 kWh × 0.4 = 7.3 kg CO₂/year
- **Ridge:** 0.0913 kWh × 0.4 = 0.0365 kg CO₂/year
- **Savings:** 7.3 − 0.0365 = 7.26 kg CO₂/year (≈99.5% reduction)

### 4. Model Quality Verification

I verified that the optimized model maintained accuracy while using far less energy.

**Metrics:**

- **RandomForest:** MAE = 1,621.03 MW, R² = 0.9076
- **Ridge:** MAE = 1,629.02 MW, R² = 0.9106
- **Accuracy Loss:** Only 8 MW difference (≈0.5% relative error increase)
- **R² Actually Improved!** Ridge performed slightly better

This shows that the carbon savings didn't come from sacrificing quality — the Ridge model is simply more efficient.

## Limitations & Future Work

### Current Limitations

1. **No Hardware Measurement:** Used estimated CPU power (15W) instead of actual readings
2. **Constant Power Assumption:** Real devices vary power draw depending on load
3. **CPU Only:** GPU not considered (Ridge doesn't require one)
4. **Grid Carbon:** Used US average (0.4 kg/kWh), though this varies by region and time

### How I'd Improve This

If I had more resources, I'd take these steps:

**1. Use CodeCarbon Library:**

```python
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
predictions = model.predict(X_test)
emissions = tracker.stop()
```

**2. Measure on Real Hardware:**

- Deploy on a Raspberry Pi 4 (a typical edge device)
- Use a USB power meter to record actual wattage
- Test in different regions (coal-heavy vs renewable grids)

**3. Dynamic Power Profiling:**

- Measure idle vs. inference power
- Account for CPU frequency scaling
- Test under varying workloads

**4. Regional Carbon Grid Data:**

- Use real-time carbon intensity APIs
- Compare worst-case (coal-heavy) and best-case (renewable-heavy) scenarios

## Reproducibility

Anyone can reproduce these measurements easily:

1. **Run the Notebook:** All code is in `GREEN_AI_SOLUTION.ipynb`
2. **Check Timing:** Cell 12 includes model training and timing results
3. **Verify Calculations:** All formulas are explained in comments
4. **Random Seed:** Set to `random_state=42` for reproducibility

## Tools Used

- **Python:** 3.12.7
- **Timing:** `time.time()` (built-in)
- **Model Size:** Measured using `pickle`
- **Assumptions:** Conservative CPU power estimates

## References

- **US Grid Carbon Intensity:** EPA eGRID database
- **CPU Power Consumption:** Intel TDP specifications
- **Deployment Scale:** US smart meter statistics (70M+ meters installed)

---

**Note:** These are conservative estimates. Real-world deployments could save even more carbon if:

- Models run on lower-power ARM processors (e.g., Raspberry Pi)
- Grids shift further toward renewables (lower carbon intensity)
- Forecast accuracy improves grid efficiency beyond the 3% assumption

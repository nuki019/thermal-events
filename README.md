# Synthetic Thermal Event Camera Benchmark (DARPA FENCE era)

**Toward a Synthetic Thermal Event Benchmark: Converting Thermal Infrared Video to Event Streams with Sensor-Physics-Aware Simulation**

This repository accompanies our pre-hardware algorithm study of event-based thermal imaging. Event-based infrared hardware is arriving (DARPA FENCE; Raytheon event-based MWIR demonstration, Apr 2026; uncooled Poisson bolometer LWIR pixels), but no public thermal event data — real or synthetic — exists. We build:

1. **ThermEv-pipeline**: the first open-source thermal-infrared → event stream conversion pipeline with thermal-specific AGC de-flickering and bolometer sensor-physics parameterization (time constant, NETD↔threshold mapping, Poisson switching noise).
2. **Synthetic thermal event detection benchmark**: thermal frames vs. thermal events vs. early fusion, with scene slicing (day/night, motion magnitude, target size).
3. **Fidelity study**: simulator × parameter grid evaluated with event statistics envelopes + downstream mAP sensitivity bands.

## Layout
- `thermal_events/` — pipeline, simulator, models, evaluation code
- `experiments/` — experiment drivers and configs
- `scripts/` — data download / utility scripts
- `paper/` — XeLaTeX paper source
- `docs/` — project planning documents
- `data/` — (git-ignored) raw and converted datasets

## License
Code: MIT. Data: subject to source dataset licenses (HIT-UAV: CC BY 4.0).

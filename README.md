# Explainable ML Visualizer

An interactive machine learning playground focused on **making models understandable, not just accurate**.
This project visualizes how common ML algorithms learn by exposing their internal mechanics in real time.

---

## Overview

Machine learning is often taught and used as a black box: data goes in, predictions come out, and the learning process stays hidden.
**Explainable ML Visualizer** takes the opposite approach.

This project provides interactive, step-by-step visualizations of foundational machine learning algorithms, allowing users to **see how models evolve as they train**, how decisions are made, and how data characteristics affect learning behavior.

The goal is to build intuition around optimization, decision boundaries, clustering, and dimensionality reduction—bridging the gap between mathematical theory and practical understanding.

---

## Features

* Real-time visualization of model training
* Interactive datasets and adjustable hyperparameters
* Clear exposure of internal model states
* Export snapshots or animated GIFs of visualizations
* Focus on interpretability and learning dynamics

### Supported / Planned Algorithms

* Linear Regression (gradient descent visualization)
* Logistic Regression / Linear Classifiers
* K-Means Clustering
* Principal Component Analysis (PCA)
* Decision Trees *(planned)*

---

## Why This Project?

Most ML tools prioritize performance metrics while hiding the learning process.
This project prioritizes **transparency, intuition, and exploration**.

It is designed for:

* Students learning machine learning concepts
* Educators looking for visual teaching aids
* Practitioners who want deeper intuition
* Anyone curious about *why* models behave the way they do

---

## Target Users & Learning Goals

Primary personas:

* Students building first-principles intuition about model behavior
* Educators designing interactive lessons or lab exercises
* Practitioners debugging or validating model dynamics
* Curious learners exploring ML fundamentals without heavy math

Learning journeys (to guide backlog prioritization):

* See gradient descent converge and relate step size to stability
* Understand decision boundaries and how they shift with data or regularization
* Visualize clustering behavior as centroids move and assignments change
* Explore dimensionality reduction by watching variance captured over time
* Inspect feature influence and model state evolution during training

---

## Tech Stack

* **Language:** Python
* **Core Libraries:** NumPy, SciPy, scikit-learn
* **Visualization:** Matplotlib (decision documented below)
* **UI / Interaction:** Lightweight interactive controls

---

## Visualization Stack

**Chosen backend:** Matplotlib (desktop/native rendering) with optional `ipywidgets` for notebook sliders.

### Required dependencies

* `matplotlib` (core plotting + animation)
* `numpy` (data generation and array operations used in live updates)
* `pillow` (image/GIF export support)

> Optional: `ipywidgets` for Jupyter-based interactivity (sliders, dropdowns). This is not required for running the core scripts.

### Rationale & constraints

* **Real-time updates:** Matplotlib supports tight training loops via `plt.pause` or `FuncAnimation` without a web server.
* **Interactivity:** Basic controls can be implemented with Matplotlib widgets or (optionally) `ipywidgets` in notebooks.
* **Export support:** Matplotlib + Pillow can export static images and animations (e.g., PNG/GIF) for teaching materials.

---

## Visualization Library Decision

**Decision: Matplotlib for the primary visualization layer.**

### Evaluation Summary (Matplotlib vs Plotly)

| Criteria | Matplotlib | Plotly (incl. Dash) |
| --- | --- | --- |
| **Real-time updates** | ✅ `plt.pause` / `FuncAnimation` for tight training loops | ✅ Streamable, but typically via Dash callbacks or Jupyter contexts |
| **Dependency footprint** | ✅ Lightweight, no web server required | ⚠️ Adds web framework/runtime (Dash) or notebook requirement |
| **Local/offline usage** | ✅ Works out of the box in local scripts | ⚠️ Best experience in browser or notebook |
| **Teaching/annotation** | ✅ Mature annotation + custom artists | ✅ Rich hover, but more web-first |
| **Integration complexity** | ✅ Simple for step-by-step training loops | ⚠️ More plumbing for state + callbacks |

### Rationale

This project prioritizes **tight, step-by-step training loops** and **minimal setup** for learners.
Matplotlib provides:

* Straightforward real-time updates without a server
* Predictable performance for incremental model visualizations
* Broad familiarity for students and educators

Plotly remains a strong option for future UI expansion (e.g., a browser-based dashboard), but the default stack will remain Matplotlib to keep the core experience lightweight and accessible.

---

## Basic Usage Patterns (Matplotlib)

### 1) Incremental training loop (real-time updates)

```python
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200)
line, = ax.plot(x, np.sin(x))
ax.set_ylim(-1.1, 1.1)

for step in range(200):
    phase = step * 0.1
    line.set_ydata(np.sin(x + phase))
    ax.set_title(f"Step {step}")
    fig.canvas.draw()
    plt.pause(0.01)
```

### 2) Structured animation (clean separation of update step)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200)
line, = ax.plot(x, np.sin(x))
ax.set_ylim(-1.1, 1.1)

def update(frame):
    line.set_ydata(np.sin(x + frame * 0.1))
    ax.set_title(f"Frame {frame}")
    return line,

ani = FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()
```

---

## Performance Note (Informal Benchmarking)

For smooth console and Matplotlib updates on a typical laptop, aim for **~50–300 samples**
per dataset and **<200 update steps**. Larger datasets (500+ points) can still work, but
updates may stutter unless you increase `display_every` or the render throttle
(`min_render_interval_s`) to reduce redraw frequency.

---

## Exporting Visualizations (PNG/GIF)

The interactive linear regression plot ships with built-in export controls:

* **Save PNG**: captures the current frame to `exports/linear_regression_XXXX.png`
* **Record GIF**: toggles frame capture and writes `exports/linear_regression.gif` when stopped

To launch the interactive view:

```python
from visualizations.linear_regression_plot import LinearRegressionInteractiveTrainer

trainer = LinearRegressionInteractiveTrainer(
    learning_rate=0.1,
    iterations_per_step=1,
    max_iterations=100,
    export_dir="exports",
)
trainer.show()
```

If you want to script exports directly, use the shared exporter utility:

```python
from utils.exporter import VisualizationFrameExporter

exporter = VisualizationFrameExporter(output_dir="exports")
exporter.save_png(fig, filename="snapshot.png")
exporter.capture_frame(fig)  # call this for each frame you want
exporter.finalize_gif(gif_name="demo.gif")
```

---

## Project Structure

```
explainable-ml-visualizer/
├── data/                # Sample or synthetic datasets
├── models/              # ML model implementations
├── visualizations/      # Visualization logic
├── utils/               # Shared utilities
├── notebooks/           # Exploratory notebooks (optional)
├── main.py              # Entry point
└── README.md
```

---

## Getting Started

### Prerequisites

* Python 3.10+
* pip

### Installation

```bash
git clone https://github.com/yourusername/explainable-ml-visualizer.git
cd explainable-ml-visualizer
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

This runs a placeholder visualization loop that prints step updates in the console.

---

## Adding New Algorithms

* Add model implementations in `models/` (e.g., training logic, parameter updates).
* Add visualization logic in `visualizations/` to render model state per step.
* Wire the model + visualization together in `main.py` or a future orchestrator module.

---

## Roadmap

* [ ] Add interactive dataset generation
* [ ] Implement decision tree visualizations
* [ ] Add robustness and noise analysis
* [ ] Improve UI responsiveness
* [ ] Export visualizations as images or GIFs

---

## Philosophy

> Accuracy tells you *what* happened.
> Explainability tells you *why*.

This project is built around the belief that understanding model behavior is just as important as optimizing metrics.

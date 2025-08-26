# fall_search (v6 stdlib)

Monte-Carlo landing dispersion simulator for falling or tumbling aerial objects under wind and gust conditions.  
Written in **stdlib-only Python**, with no external dependencies.

---

## Features

- **Physics-based simulation**
  - Semi-implicit Euler integration with exponential atmosphere model
  - Monte-Carlo sampling of drag, area, wind, and tumble parameters
  - Support for pre-release glide or time offset

- **Outputs**
  - **CSV** of landing points
  - **JSON** summary with statistics and radii
  - **GeoJSON + KML** search polygons (histogram & KDE/HDR/KDR)
  - **Distance-first containment radii** (mode, centroid, release)
  - **Archimedean spiral search path**

- **Config system**
  - JSON configuration loader (backwards-compatible with v5 schema)
  - Input validation and deep-merge of presets
  - Optional integration with [Open-Meteo API](https://open-meteo.com/) for current wind data

- **Portable**
  - Pure Python, no external packages
  - Can run natively (CLI) or entirely in the browser via [Pyodide](https://pyodide.org/) and GitHub Pages

---

## Getting Started

### Prerequisites
- Python 3.8+ (stdlib only)

### Running a Simulation
Clone the repository:

```bash
git clone https://github.com/<your-username>/fall-search.git
cd fall-search

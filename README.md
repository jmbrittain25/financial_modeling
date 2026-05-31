# Financial Simulator

A professional-grade, extensible platform for stochastic financial modeling and Monte Carlo analysis.

Built from the ground up to be powerful for humans *and* future autonomous AI agents.

## Features

- **Flexible Discrete-Event Engine** — Model complex real-world cash flows with timing, growth, stochastic behavior, loans, taxes, investments, and more.
- **Rich Distributions & Stochastic Processes** — Normal, LogNormal, Beta, Triangular, Geometric Brownian Motion, Mean Reversion, etc.
- **Risk & Quantitative Analytics** — VaR, CVaR, Sharpe, Sortino, Max Drawdown, Probability of Ruin, and more.
- **Monte Carlo at Scale** — Efficient parallel execution of thousands of simulations.
- **Modern Interfaces** — Typer CLI + FastAPI backend + Streamlit prototype.
- **AI-Agent Ready** — Clean, typed, serializable interfaces designed for programmatic and LLM-driven use.

## Quick Start

### CLI

```bash
# Run a simulation from config
simulate run --config my-config.yaml --seed 42 --verbose

# Or via Python
python -m financial_simulator run -c my-config.yaml --seed 42
```

### Python API

```python
from financial_simulator.core import SimulationEngine
from examples.retirement import create_retirement_engine

engine = create_retirement_engine(seed=42)
engine.run()
result = engine.get_result()
print(result.final_state)
```

### Web Interface (Streamlit) — Full Interactive Scenario Builder

```bash
pip install -e .
streamlit run app/streamlit_app.py
```

The app is now a powerful **no-code decision-support tool**:

**Key Features**
- **Interactive Scenario Builder**: Visually build complex scenarios using 8+ realistic one-click presets, full timing + value generator editors, external drivers, continuous processes, and custom metrics.
- **Live Distribution Editor**: 7 distribution types with real-time Plotly previews + analytical PDFs.
- **Rich Template Gallery**: 5 high-quality curated starting scenarios (Retirement, Real Estate, Business, Tax Planning, Portfolio).
- **Scenario Overview**: Visual event schedule preview + complexity indicators before you run.
- **Powerful Results Dashboard**: Percentile fan charts over time, custom metric analysis, risk metrics, and downloads.
- **Personal Libraries**: Save and reload your own scenarios and custom distributions (persisted to disk).
- **Save/Load**: Full import/export + personal library support.

Everything is built on the strong declarative `ScenarioConfig` model so your work is fully serializable and runnable from code or the CLI as well.

## Architecture

The platform is deliberately modular:

- `core/` — SimulationEngine, Event system, Distributions, Stochastic Processes, Financial Models
- `analytics/` — Risk metrics and Monte Carlo analysis
- `monte_carlo/` — Parallel execution runner
- `api/` — FastAPI backend
- `app/` — Web interfaces (Streamlit prototype)

## Example Scenarios

Located in `examples/`:

- `retirement.py` — 30-year retirement planning with contributions + growth
- `business_cashflow.py` — Small business revenue and expense modeling

More examples are being added continuously.

## Design Philosophy

This project is built with the following principles:

1. **Extensibility first** — New financial concepts should be easy to add.
2. **Reproducibility** — Every simulation is fully deterministic given a seed.
3. **AI Agent Friendly** — Structured I/O, clear schemas, minimal magic.
4. **Practical power** — Balance modeling fidelity with usability.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for long-term vision, including autonomous AI agents that can design and iteratively improve Monte Carlo simulations.

## Development

```bash
pip install -e .
```

Run tests / CLI / web interface as described above.

## License

MIT (or similar — to be finalized).

---

**Built with the belief that the future of financial decision-making will be heavily augmented by autonomous simulation agents.**

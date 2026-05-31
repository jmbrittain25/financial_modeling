"""
External Drivers — usage examples.

Run:
    python examples/external_drivers.py
"""

from datetime import datetime, timedelta

from financial_simulator.core.event import ComposedEventBuilder, FixedValue, IntervalTiming
from financial_simulator.scenarios import (
    ScenarioConfig,
    make_inflation_driver,
    make_interest_rate_driver,
    make_stock_market_driver,
    run_single,
    sample_driver_path,
)


def demo_sampling():
    print("=== Sampling External Drivers (no simulation required) ===")
    start = datetime(2026, 1, 1)
    end = datetime(2027, 1, 1)

    rate_d = make_interest_rate_driver()
    p = sample_driver_path(rate_d, start, end, freq="QS", n_paths=3, seed=42)
    print(f"Interest rate driver: {rate_d.name} → {rate_d.target_state_key}")
    print(f"  Sampled {len(p['times'])} points across {len(p['paths'])} paths")
    print(f"  Terminal stats: mean={p['summary']['mean_terminal']:.4f}")

    inf_d = make_inflation_driver()
    p2 = sample_driver_path(inf_d, start, end, freq="M", n_paths=1, seed=7)
    print(f"Inflation driver terminal mean: {p2['summary']['mean_terminal']:.4f}")

    mkt_d = make_stock_market_driver(initial_value=100.0)
    p3 = sample_driver_path(mkt_d, start, end, freq="MS", n_paths=2, seed=99)
    print(f"Equity market (GBM) paths generated. Final values: {[round(v,1) for v in p3['paths'][0][-3:]]}")


def demo_full_scenario_with_drivers():
    print("\n=== Full Scenario with 3 External Drivers ===")
    start = datetime(2026, 1, 1)
    end = datetime(2028, 1, 1)

    cfg = ScenarioConfig(
        name="Multi-Driver Demo",
        start=start,
        end=end,
        initial_state={"cash": 10000.0, "portfolio": 250000.0, "inflation": 1.0},
        external_drivers=[
            make_interest_rate_driver(target_state_key="mortgage_rate"),
            make_inflation_driver(target_state_key="inflation"),
            make_stock_market_driver(target_state_key="portfolio", initial_value=250000.0),
        ],
        # Dummy monthly events are required so the simulation clock advances.
        # Without any events, ContinuousProcess objects (from GBM/mean-revert drivers) never get a chance to call .advance().
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30), start_time=start),
                value_gen=FixedValue(value=0.0),
            )
        ],
    )

    res = run_single(cfg, seed=123)
    print("Run complete.")
    print("Final state keys affected by drivers:", {k: round(v, 2) if isinstance(v, (int, float)) else v
                                                    for k, v in res.final_state.items() if k in ("mortgage_rate", "inflation", "portfolio")})


if __name__ == "__main__":
    demo_sampling()
    demo_full_scenario_with_drivers()
    print("\nDone. External drivers are fully operational as first-class objects.")

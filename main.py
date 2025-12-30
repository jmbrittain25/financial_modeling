from typing import Dict, Any
import datetime as dt

import financial_simulator as fs


def real_estate_factory(params: Dict[str, Any]) -> fs.Simulation:
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2030, 12, 31)  # 5-year horizon
    appreciation_rate = params.get('appreciation_rate', 0.04)
    sim = fs.Simulation("RealEstateSim", start, end, appreciation_rate)
    
    # Property appraisal and purchase
    appraisal = params.get('appraisal', 300000.0)
    down_fraction = params.get('down_fraction', 0.5)
    down_payment = appraisal * down_fraction
    seller_principal = appraisal - down_payment
    closing_fees = params.get('closing_fees', 10000.0)
    purchase_value = - (down_payment + closing_fees)
    sim.add_builder(fs.TriggeredEventBuilder(start, purchase_value, metadata={'type': 'purchase'}))  # One-time at start
    
    # HELOC: Variable rate, amount withdrawn as down payment
    heloc_draw = params.get('heloc_draw', down_payment)  # Could be less if cash used
    heloc_rate_dist = fs.NormalDistribution(params.get('heloc_initial_rate', 0.075), 0.005)  # Varies quarterly
    heloc_builder = fs.VariableRateLoanBuilder(
        heloc_draw, params.get('heloc_initial_rate', 0.075), 120, start,
        heloc_rate_dist, dt.timedelta(days=90), {'type': 'heloc'}
    )
    sim.add_builder(heloc_builder)
    
    # Seller financing
    seller_rate = params.get('seller_rate', 0.05)
    seller_term_months = int(params.get('seller_term_months', 120))
    seller_builder = fs.VariableRateLoanBuilder(
        seller_principal, seller_rate, seller_term_months, start,
        fs.NormalDistribution(seller_rate, 0.005), dt.timedelta(days=365), {'type': 'seller_financing'}  # Annual change
    )
    sim.add_builder(seller_builder)
    
    # Rent: Growing, starting after purchase
    rent_start = start + dt.timedelta(days=30)
    rent_builder = fs.GrowingValueGenerator(
        params.get('monthly_rent', 2000.0), params.get('rent_growth', 0.03), dt.timedelta(days=30),
        {'type': 'rent_income'}, start_time=rent_start
    )
    sim.add_builder(rent_builder)
    
    # Maintenance: Seasonal (lawn summer), fixed (other), unexpected repairs (stochastic, say annual)
    lawn_builder = fs.SeasonalEventBuilder(
        params.get('monthly_lawn', -100.0), dt.timedelta(days=30), [5,6,7,8,9], {'type': 'lawn'}
    )
    sim.add_builder(lawn_builder)
    other_maint_builder = fs.FixedValueGenerator(params.get('monthly_maint', -200.0), dt.timedelta(days=30), {'type': 'maintenance'})
    sim.add_builder(other_maint_builder)
    unexpected_builder = fs.GrowingValueGenerator(  # Inflating 3%/year
        -appraisal * 0.01, 0.03, dt.timedelta(days=365), {'type': 'unexpected_repairs'}
    )
    sim.add_builder(unexpected_builder)
    
    # Mom leave time: Triggers post-leave renovations
    mom_leave = params.get('mom_leave_time', start + dt.timedelta(days=730))  # ~2 years min
    # Pre-leave renovations: e.g., kitchen immediate
    kitchen_cost = params.get('kitchen_cost', -30000.0)
    sim.add_builder(fs.TriggeredEventBuilder(start + dt.timedelta(days=90), kitchen_cost, metadata={'type': 'kitchen_renov'}))
    
    # Post-leave: Floors, central air
    floors_cost = params.get('floors_cost', -10000.0)
    sim.add_builder(fs.TriggeredEventBuilder(mom_leave, floors_cost, dt.timedelta(days=30), {'type': 'floors_renov'}))
    central_air_cost = params.get('central_air_cost', -8000.0)
    sim.add_builder(fs.TriggeredEventBuilder(mom_leave, central_air_cost, dt.timedelta(days=60), {'type': 'central_air_renov'}))
    
    # Property appreciation tracked in run()
    # Other: Inflation on all costs implicitly via growing generators
    
    return sim

if __name__ == "__main__":

    # Distributions based on research (realistic for 2025 Mechanicsburg PA scenario)
    dists = {
        'heloc_draw': fs.UniformDistribution(100000, 150000),  # Amount withdrawn
        'heloc_initial_rate': fs.NormalDistribution(0.075, 0.005),  # ~7.5% avg, variable
        'down_fraction': fs.UniformDistribution(0.4, 0.6),  # 40-60% down
        'appraisal': fs.NormalDistribution(300000, 20000),  # ~$300k avg
        'closing_fees': fs.TriangularDistribution(5000, 10000, 16000),  # 2-5% of $300k (positive, negated in factory)
        'seller_rate': fs.UniformDistribution(0.04, 0.06),  # Family deal 4-6%
        'seller_term_months': fs.UniformDistribution(60, 120),  # 5-10 years
        'kitchen_cost': fs.TriangularDistribution(-75000, -40000, -25000),  # Renov costs (negative for expense)
        'floors_cost': fs.TriangularDistribution(-15000, -10000, -5000),
        'central_air_cost': fs.TriangularDistribution(-10000, -7000, -5000),
        'monthly_rent': fs.NormalDistribution(2000, 200),  # ~$2000 avg
        'monthly_lawn': fs.NormalDistribution(-50, 10),  # Per month in season
        'monthly_maint': fs.NormalDistribution(-200, 50),  # General
        'rent_growth': fs.NormalDistribution(0.03, 0.01),  # 3% annual
        'mom_leave_time': fs.DateDistribution(  # At least 2 years, up to 5
            dt.datetime(2026, 1, 1) + dt.timedelta(days=730),
            dt.datetime(2026, 1, 1) + dt.timedelta(days=1825)
        ),
        'appreciation_rate': fs.NormalDistribution(0.04, 0.005)  # 4% annual ±0.5%
    }

    # Build and run 100 Monte Carlo simulations
    builder = fs.SimulationBuilder(real_estate_factory, dists)
    sims = builder.build_simulations(1_000, seed=42)  # Use seed for repeatability

    # Analyze: Stats, plots for cash, property value, distributions
    analyzer = fs.SimulationAnalyzer(sims)
    print(analyzer.compute_statistics())
    analyzer.plot_cumulative_cash_flows()
    analyzer.plot_property_values()
    analyzer.plot_histogram_end_values()

    # Serialize one for example
    sims[0].save_json("example_sim.json")

    print("Simulations complete")

import json
import datetime as dt


def generate_config(output_file='config.json'):
    dists = {
        'heloc_draw': {'type': 'UniformDistribution', 'low': 100000, 'high': 150000},
        'heloc_initial_rate': {'type': 'NormalDistribution', 'mean': 0.075, 'std': 0.005},
        'down_fraction': {'type': 'UniformDistribution', 'low': 0.4, 'high': 0.6},
        'appraisal': {'type': 'NormalDistribution', 'mean': 300000, 'std': 20000},
        'closing_fees': {'type': 'TriangularDistribution', 'low': 5000, 'mode': 10000, 'high': 16000},
        'seller_rate': {'type': 'UniformDistribution', 'low': 0.04, 'high': 0.06},
        'seller_term_months': {'type': 'UniformDistribution', 'low': 60, 'high': 120},
        'kitchen_cost': {'type': 'TriangularDistribution', 'low': -75000, 'mode': -40000, 'high': -25000},
        'floors_cost': {'type': 'TriangularDistribution', 'low': -15000, 'mode': -10000, 'high': -5000},
        'central_air_cost': {'type': 'TriangularDistribution', 'low': -10000, 'mode': -7000, 'high': -5000},
        'monthly_rent': {'type': 'NormalDistribution', 'mean': 2000, 'std': 200},
        'monthly_lawn': {'type': 'NormalDistribution', 'mean': -50, 'std': 10},
        'monthly_maint': {'type': 'NormalDistribution', 'mean': -200, 'std': 50},
        'rent_growth': {'type': 'NormalDistribution', 'mean': 0.03, 'std': 0.01},
        'mom_leave_time': {
            'type': 'DateDistribution',
            'start': (dt.datetime(2026, 1, 1) + dt.timedelta(days=730)).isoformat(),
            'end': (dt.datetime(2026, 1, 1) + dt.timedelta(days=1825)).isoformat()
        },
        'appreciation_rate': {'type': 'NormalDistribution', 'mean': 0.04, 'std': 0.005},
    }
    
    simulation = {
        'start': dt.datetime(2026, 1, 1).isoformat(),
        'end': dt.datetime(2036, 1, 1).isoformat(),
        'initial_state': {
            'property_value': '${appraisal}',
            'cumulative_cash': 0.0,
            'heloc_rate': '${heloc_initial_rate}',
            'seller_rate': '${seller_rate}'
        },
        'continuous_processes': [
            {'type': 'Appreciation', 'rate': '${appreciation_rate}', 'var': 'property_value'}
        ],
        'builders': [
            # Purchase net cash out (negative)
            {'name': 'purchase', 'timing': {'type': 'OneTime', 'time': dt.datetime(2026, 1, 1).isoformat()},
             'value_gen': {'type': 'Fixed', 'value': '${purchase_cash}'},
             'metadata': {'type': 'purchase'}},
            # HELOC draw (positive cash in)
            {'name': 'heloc_draw', 'timing': {'type': 'OneTime', 'time': dt.datetime(2026, 1, 1).isoformat()},
             'value_gen': {'type': 'Fixed', 'value': '${heloc_draw}'},
             'metadata': {'type': 'heloc_draw'}},
            # HELOC payments (monthly, variable rate)
            {'name': 'heloc_payment', 'timing': {'type': 'Interval', 'interval_days': 30, 'start_time': dt.datetime(2026, 2, 1).isoformat()},
             'value_gen': {'type': 'VariableRateLoan', 'principal': '${heloc_principal}', 'initial_rate': '${heloc_initial_rate}', 'term_months': 360, 'rate_key': 'heloc_rate'},
             'metadata': {'type': 'heloc'}},
            # Seller financing payments
            {'name': 'seller_payment', 'timing': {'type': 'Interval', 'interval_days': 30, 'start_time': dt.datetime(2026, 2, 1).isoformat()},
             'value_gen': {'type': 'VariableRateLoan', 'principal': '${seller_principal}', 'initial_rate': '${seller_rate}', 'term_months': '${seller_term_months}', 'rate_key': 'seller_rate'},
             'metadata': {'type': 'seller_financing'}},
            # Rent income (monthly, growing)
            {'name': 'rent', 'timing': {'type': 'Interval', 'interval_days': 30, 'start_time': dt.datetime(2026, 1, 1).isoformat()},
             'value_gen': {'type': 'Growing', 'initial': '${monthly_rent}', 'growth_rate': '${rent_growth}'},
             'metadata': {'type': 'rent_income'}},
            # Lawn expense (seasonal, summer months)
            {'name': 'lawn', 'timing': {'type': 'Seasonal', 'months': [6,7,8,9], 'inner': {'type': 'Interval', 'interval_days': 30, 'start_time': dt.datetime(2026, 1, 1).isoformat()}},
             'value_gen': {'type': 'Fixed', 'value': '${monthly_lawn}'},
             'metadata': {'type': 'lawn'}},
            # Maintenance
            {'name': 'maintenance', 'timing': {'type': 'Interval', 'interval_days': 30, 'start_time': dt.datetime(2026, 1, 1).isoformat()},
             'value_gen': {'type': 'Fixed', 'value': '${monthly_maint}'},
             'metadata': {'type': 'maintenance'}},
            # Unexpected repairs (random 5 times over period) — hardcoded dist
            {'name': 'unexpected_repairs', 'timing': {'type': 'Random', 'start': dt.datetime(2026, 1, 1).isoformat(), 'end': dt.datetime(2036, 1, 1).isoformat(), 'n': 5},
             'value_gen': {'type': 'Distribution', 'dist': {'type': 'TriangularDistribution', 'low': -5000, 'mode': -2000, 'high': -1000}},
             'metadata': {'type': 'unexpected_repairs'}},
            # Renovations (one-time)
            {'name': 'kitchen_renov', 'timing': {'type': 'OneTime', 'time': dt.datetime(2026, 6, 1).isoformat()},
             'value_gen': {'type': 'Fixed', 'value': '${kitchen_cost}'},
             'metadata': {'type': 'kitchen_renov'}},
            {'name': 'floors_renov', 'timing': {'type': 'OneTime', 'time': dt.datetime(2026, 7, 1).isoformat()},
             'value_gen': {'type': 'Fixed', 'value': '${floors_cost}'},
             'metadata': {'type': 'floors_renov'}},
            {'name': 'central_air_renov', 'timing': {'type': 'OneTime', 'time': dt.datetime(2026, 8, 1).isoformat()},
             'value_gen': {'type': 'Fixed', 'value': '${central_air_cost}'},
             'metadata': {'type': 'central_air_renov'}},
            # HELOC rate change (yearly) — hardcoded dist for per-event sampling
            {'name': 'heloc_rate_change', 'timing': {'type': 'Interval', 'interval_days': 365, 'start_time': dt.datetime(2027, 1, 1).isoformat()},
             'value_gen': {'type': 'RateChange', 'dist': {'type': 'NormalDistribution', 'mean': 0.075, 'std': 0.01}, 'update_key': 'heloc_rate'},
             'metadata': {'type': 'rate_change'}},
            # Example event for mom_leave_time
            {'name': 'mom_leave', 'timing': {'type': 'OneTime', 'time': '${mom_leave_time}'},
             'value_gen': {'type': 'Fixed', 'value': 0.0},
             'metadata': {'type': 'mom_leave'}},
        ]
    }
    
    config = {
        'num_simulations': 1000,
        'seed': 42,
        'dists': dists,
        'simulation': simulation
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Config saved to {output_file}")

if __name__ == "__main__":
    generate_config()
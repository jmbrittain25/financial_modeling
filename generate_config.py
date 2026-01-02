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
        'appreciation_rate': {'type': 'NormalDistribution', 'mean': 0.04, 'std': 0.005}
    }
    
    config = {
        'num_simulations': 1000,
        'seed': 42,
        'dists': dists
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Config saved to {output_file}")

if __name__ == "__main__":
    generate_config()

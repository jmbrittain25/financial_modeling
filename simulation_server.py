from typing import Dict, Any
from flask import Flask, request, jsonify
import threading
import uuid
import datetime as dt

import financial_simulator as fs


app = Flask(__name__)

# In-memory storage for jobs (for MVP)
jobs = {}  # {job_id: {'status': 'running/completed/failed', 'progress': 0, 'message': '', 'results': {}}}

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
    sim.add_builder(fs.TriggeredEventBuilder(start, heloc_draw, metadata={'type': 'heloc_draw'}))  # Add initial inflow
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
    
    sim.params = params  # Store parameters for analysis
    sim.initial_property_value = appraisal  # Set initial value
    
    return sim

def run_simulations(config):
    job_id = config['job_id']
    try:
        # Recreate dists from JSON
        param_distributions = {k: fs.create_distribution(v) for k, v in config['dists'].items()}
        
        # Build and run
        builder = fs.SimulationBuilder(real_estate_factory, param_distributions)
        sims = builder.build_simulations(config['num_simulations'], seed=config['seed'])
        
        # Analyze
        analyzer = fs.SimulationAnalyzer(sims)
        stats = analyzer.compute_statistics()
        
        # Prepare results: Serialize sims, events, metrics
        results = {
            'sims': [sim.to_dict() for sim in sims],  # Includes events, state_history
            'stats': stats,  # Global metrics
            # Add time-series metrics if needed, e.g., from analyzer.to_dataframe(sims[0])
        }
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Done'
        jobs[job_id]['results'] = results
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = str(e)

@app.route('/simulate', methods=['POST'])
def simulate():
    config = request.json
    job_id = config.get('job_id', str(uuid.uuid4()))
    jobs[job_id] = {'status': 'running', 'progress': 0, 'message': 'Starting', 'results': None}
    
    # Start background thread
    thread = threading.Thread(target=run_simulations, args=(config,))
    thread.start()
    
    return jsonify({'job_id': job_id}), 202

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id, {'status': 'not_found', 'progress': 0, 'message': 'Job not found'})
    return jsonify(job)

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    job = jobs.get(job_id)
    if job and job['status'] == 'completed':
        return jsonify(job['results'])
    return jsonify({'error': 'Results not ready or not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

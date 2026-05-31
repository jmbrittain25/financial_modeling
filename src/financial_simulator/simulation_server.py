from typing import Dict, Any
from flask import Flask, request, jsonify
import threading
import uuid
import datetime as dt
import copy
import traceback  # Add at the top of the file

import financial_simulator as fs
from financial_simulator.sim_builder import _build_one


app = Flask(__name__)

jobs = {}


def substitute(config: Any, params: Dict) -> Any:
    if isinstance(config, dt.datetime):
        return config.isoformat()
    if isinstance(config, dict):
        return {k: substitute(v, params) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute(v, params) for v in config]
    elif isinstance(config, str):
        if config.startswith('-${') and config.endswith('}'):
            key = config[3:-1]
            val = substitute(params.get(key), params)
            if not isinstance(val, (int, float)):
                raise ValueError(f"Cannot negate non-numeric value for key '{key}'")
            return -val
        elif config.startswith('${') and config.endswith('}'):
            key = config[2:-1]
            return substitute(params.get(key), params)
    return config


def create_simulation(params: Dict, base_config: Dict) -> fs.Simulation:
    # Compute derived parameters
    params['down_amount'] = params['appraisal'] * params['down_fraction']
    params['closing_cost'] = -params['closing_fees']
    params['purchase_cash'] = params['closing_cost'] - params['down_amount']  # Negative cash out
    params['seller_principal'] = params['appraisal'] - params['down_amount'] - params['heloc_draw']
    if params['seller_principal'] < 0:
        params['seller_principal'] = 0  # Avoid negative principal
    params['heloc_principal'] = params['heloc_draw']

    sub_config = substitute(copy.deepcopy(base_config['simulation']), params)
    start = dt.datetime.fromisoformat(sub_config['start'])
    end = dt.datetime.fromisoformat(sub_config['end'])
    sim = fs.Simulation(sub_config.get('name', 'GeneralSim'), start, end, params)
    sim.state = sub_config.get('initial_state', {})
    for proc_d in sub_config.get('continuous_processes', []):
        sim.add_continuous(fs.create_continuous_process(proc_d))
    for builder_d in sub_config.get('builders', []):
        sim.add_builder(fs.create_event_builder(builder_d))
    return sim


def run_simulations(config):
    job_id = config['job_id']
    try:
        param_distributions = {k: fs.create_distribution(v) for k, v in config['dists'].items()}
        factory = lambda p: create_simulation(p, config)
        num = config['num_simulations']
        seed = config['seed']
        sims = []
        for i in range(num):
            sim = _build_one(factory, param_distributions, seed, i)
            sims.append(sim)
            jobs[job_id]['progress'] = int(100 * (i + 1) / num)
            jobs[job_id]['message'] = f'Completed {i + 1} of {num} simulations'
        
        # New: Signal analysis start
        jobs[job_id]['message'] = 'Simulations complete. Analyzing results (computing stats, IRR, ROI)...'
        
        analyzer = fs.SimulationAnalyzer(sims)
        stats = analyzer.compute_statistics()
        results = {
            'sims': [sim.to_dict() for sim in sims],
            'stats': stats,
        }
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Done'
        jobs[job_id]['results'] = results
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = traceback.format_exc()  # Full stack trace


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
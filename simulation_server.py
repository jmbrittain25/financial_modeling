from typing import Dict, Any
from flask import Flask, request, jsonify
import threading
import uuid
import datetime as dt
import copy

import financial_simulator as fs


app = Flask(__name__)

jobs = {}


def substitute(config: Any, params: Dict) -> Any:
    if isinstance(config, dict):
        return {k: substitute(v, params) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute(v, params) for v in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        key = config[2:-1]
        return params.get(key)
    return config


def create_simulation(params: Dict, base_config: Dict) -> fs.Simulation:
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
        builder = fs.SimulationBuilder(lambda p: create_simulation(p, config), param_distributions)
        sims = builder.build_simulations(config['num_simulations'], seed=config['seed'])
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
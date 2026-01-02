import requests
import json
import time
import uuid
import matplotlib.pyplot as plt

import financial_simulator as fs


def send_and_monitor(server_url, config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    job_id = str(uuid.uuid4())  # Unique job ID
    config['job_id'] = job_id
    
    # Send POST
    response = requests.post(f"{server_url}/simulate", json=config)
    if response.status_code != 202:
        print(f"Error: {response.text}")
        return
    
    print(f"Job {job_id} submitted.")
    
    # Poll status
    while True:
        status_res = requests.get(f"{server_url}/status/{job_id}")
        status = status_res.json()
        print(f"Status: {status['progress']}% complete. Message: {status['message']}")
        if status['status'] == 'completed':
            break
        elif status['status'] == 'failed':
            print(f"Failed: {status['message']}")
            return
        time.sleep(5)
    
    # Retrieve results
    results_res = requests.get(f"{server_url}/results/{job_id}")
    results = results_res.json()
    
    # Process and plot (adapt from your sim_analyzer.py)
    # Assuming results include list of sim dicts; recreate Simulations
    sims = [fs.Simulation.from_dict(sim) for sim in results['sims']]
    analyzer = fs.SimulationAnalyzer(sims)
    stats = analyzer.compute_statistics()
    print("Global Metrics:", stats)
    
    # Plots (e.g., net worth, histogram)
    analyzer.plot_net_worth(title="Demo Net Worth Percentiles")
    analyzer.plot_histogram_end_values(title="Demo Ending Net Worth Distribution")
    plt.show()  # Or save to file

if __name__ == "__main__":
    server_url = input("Enter server URL (e.g., https://your-ngrok-url.ngrok.io): ")
    send_and_monitor(server_url)
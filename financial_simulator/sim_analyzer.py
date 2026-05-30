from typing import List, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton

from .sim import Simulation


def get_color(frac: float):
    return plt.cm.viridis(frac)


class SimulationAnalyzer:
    def __init__(self, sims: List[Simulation]):
        self.sims = sims

    def to_dataframe(self, sim: Simulation) -> pd.DataFrame:
        # Create pivot for categorized cash flows
        event_data = [{'time': e.time, 'value': e.value, 'type': e.metadata.get('type', 'other'),
                       'principal_pay': e.metadata.get('principal', 0.0),
                       'remaining_balance': e.metadata.get('remaining_balance', None)} for e in sim.events]
        df = pd.DataFrame(event_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        type_sum = df.pivot_table(values='value', index='time', columns='type', aggfunc='sum', fill_value=0.0)
        
        result = type_sum.copy()
        result['cash_flow'] = type_sum.sum(axis=1)
        result['cumulative_cash'] = result['cash_flow'].cumsum()  # Renamed for clarity
        
        # Join property value (align indices)
        prop_df = pd.DataFrame.from_dict(sim.state_history, orient='index')['property_value']
        prop_df.index = pd.to_datetime(prop_df.index)
        result = result.join(prop_df, how='outer').ffill().bfill()  # Fill to cover all times
        
        # Infer loan balances (with initial principal)
        loan_types = ['heloc', 'seller_financing']
        seen = set()
        for lt in loan_types:
            lt_events = df[df['type'] == lt]
            if not lt_events.empty:
                first_idx = lt_events.index[0]
                first_row = lt_events.iloc[0]
                initial_principal = first_row['remaining_balance'] + first_row['principal_pay']
                result.at[sim.start, f'{lt}_balance'] = initial_principal
                for idx, row in lt_events.iterrows():
                    result.at[idx, f'{lt}_balance'] = row['remaining_balance']
                result[f'{lt}_balance'] = result[f'{lt}_balance'].ffill().bfill()  # Handle gaps
            else:
                result[f'{lt}_balance'] = 0.0
        
        result['total_loans'] = result[[f'{lt}_balance' for lt in loan_types]].sum(axis=1)
        
        # Derived metrics
        op_ex_types = ['lawn', 'maintenance', 'unexpected_repairs']
        capex_types = ['kitchen_renov', 'floors_renov', 'central_air_renov']
        debt_types = ['heloc', 'seller_financing']
        result['revenue'] = result.get('rent_income', 0.0)
        result['operating_expenses'] = result[ [t for t in op_ex_types if t in result.columns] ].sum(axis=1)
        result['capex'] = result[ [t for t in capex_types if t in result.columns] ].sum(axis=1)
        result['debt_service'] = result[ [t for t in debt_types if t in result.columns] ].sum(axis=1)
        result['noi'] = result['revenue'] + result['operating_expenses']  # Expenses negative
        result['net_cash_flow'] = result['noi'] + result['debt_service'] + result['capex'] + result.get('other', 0.0)  # Approx equals cash_flow
        result['net_worth'] = result['cumulative_cash'] + result['property_value'] - result['total_loans']
        result['dscr'] = -result['noi'] / -result['debt_service'] if result['debt_service'].any() < 0 else np.nan  # Debt Service Coverage Ratio
        return result

    def compute_statistics(self) -> Dict:
        dfs = [self.to_dataframe(sim) for sim in self.sims]
        metrics = {
            'ending_net_worth': [df['net_worth'][-1] for df in dfs],
            'irr': [self.compute_irr(sim) for sim in self.sims],
            'roi': [self.compute_roi(sim) for sim in self.sims],
            'cap_rate': [self.compute_cap_rate(sim) for sim in self.sims],
        }
        stats = {k: {'mean': np.mean(v), 'std': np.std(v), 'percentiles': np.percentile(v, [10,50,90])} for k, v in metrics.items()}
        return stats

    def compute_irr(self, sim: Simulation) -> float:
        df = self.to_dataframe(sim)
        cfs = df['cash_flow'].values
        years = [(t - df.index[0]).days / 365.25 for t in df.index]
        def npv(r):
            return sum(cf / (1 + r)**y for y, cf in zip(years, cfs))
        try:
            return newton(npv, 0.1)
        except:
            return np.nan

    def compute_roi(self, sim: Simulation) -> float:
        df = self.to_dataframe(sim)
        initial_investment = -df['cash_flow'][df['type'] == 'purchase'].sum() if 'purchase' in df.columns else 0
        ending_cash = df['cumulative_cash'][-1]
        return (ending_cash - initial_investment) / initial_investment if initial_investment != 0 else np.nan

    def compute_cap_rate(self, sim: Simulation) -> float:
        df = self.to_dataframe(sim)
        avg_noi = df['noi'].mean()
        avg_property_value = df['property_value'].mean()
        return avg_noi / avg_property_value if avg_property_value != 0 else np.nan

    def plot_net_worth(self, title: str = "Net Worth Percentiles"):
        dfs = [self.to_dataframe(sim) for sim in self.sims]
        endings = [df['net_worth'][-1] for df in dfs]
        sorted_indices = np.argsort(endings)
        positions = np.linspace(0, len(self.sims) - 1, 11, dtype=int)
        selected_indices = sorted_indices[positions]

        min_x = min(df.index.min() for df in dfs)
        max_x = max(df.index.max() for df in dfs)

        plt.figure(figsize=(12, 6))
        for df in dfs:
            plt.plot(df.index, df['net_worth'], color='black', alpha=0.1)

        for i in reversed(range(11)):
            idx = selected_indices[i]
            percentile = i * 10
            label = f"{percentile}th percentile ({self.sims[idx].name})"
            color = get_color(i / 10.0)
            plt.plot(dfs[idx].index, dfs[idx]['net_worth'], color=color, label=label)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Net Worth")
        plt.xlim(min_x, max_x)
        plt.margins(x=0)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_histogram_end_values(self, title: str = "Distribution of Ending Net Worth"):
        endings = [self.to_dataframe(sim)['net_worth'][-1] for sim in self.sims]
        plt.figure(figsize=(10, 5))
        plt.hist(endings, bins=20, edgecolor='black')
        plt.title(title)
        plt.xlabel("Ending Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def compare_simulations(self, sim1: Simulation, sim2: Simulation, title: str = "Comparison of Cumulative Cash Flows"):
        plt.figure(figsize=(12, 6))
        df1 = self.to_dataframe(sim1)
        df2 = self.to_dataframe(sim2)
        plt.plot(df1.index, df1['cumulative_cash'], label=sim1.name)
        plt.plot(df2.index, df2['cumulative_cash'], label=sim2.name)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative Cash")
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_params(self) -> pd.DataFrame:
        # Correlate metrics with params for optimization
        data = []
        for sim in self.sims:
            row = sim.params.copy()
            df = self.to_dataframe(sim)
            row['ending_net_worth'] = df['net_worth'][-1]
            row['irr'] = self.compute_irr(sim)
            row['roi'] = self.compute_roi(sim)
            data.append(row)
        param_df = pd.DataFrame(data)
        print("Correlations with Ending Net Worth:\n", param_df.corr()['ending_net_worth'].sort_values(ascending=False))
        # Optimal: sim with max mean-adjusted (e.g., mean / std for risk-adjusted)
        optimal_idx = param_df['ending_net_worth'].idxmax()
        print("Optimal Sim Params:", param_df.iloc[optimal_idx])
        return param_df

    @classmethod
    def from_directory(cls, dir_path: str, file_extension: str = '.json') -> 'SimulationAnalyzer':
        sims = []
        for filename in os.listdir(dir_path):
            if filename.endswith(file_extension):
                filepath = os.path.join(dir_path, filename)
                if file_extension == '.json':
                    sim = Simulation.load_json(filepath)
                elif file_extension == '.pkl':
                    sim = Simulation.load_pickle(filepath)
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")
                sims.append(sim)
        return cls(sims)
from typing import List, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton

from .sim import Simulation


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
        result['dscr'] = -result['noi'] / -result['debt_service'] if result['debt_service'].any() < 0 else np.nan  # Debt Service Coverage Ratio (>1.2 ideal)
        
        return result

    def compute_irr(self, sim: Simulation, selling_cost_rate: float = 0.06) -> float:
        df = self.to_dataframe(sim)
        # Group cash flows annually
        df_resampled = df.resample('Y').sum()
        years = df_resampled.index.year
        cash_flows = df_resampled['net_cash_flow'].values
        # Initial is first year's (often negative)
        irr_series = list(cash_flows[:-1])  # Intermediate years
        # Terminal: last CF + net proceeds (prop - loans - selling costs)
        terminal = cash_flows[-1] + df['property_value'][-1] * (1 - selling_cost_rate) - df['total_loans'][-1]
        irr_series.append(terminal)
        # Use XIRR approximation
        dates = df_resampled.index
        ordinal_dates = [(d - dates[0]).days / 365.0 for d in dates]
        def npv(r):
            return sum(cf / (1 + r)**t for cf, t in zip(irr_series, ordinal_dates))
        try:
            return newton(npv, 0.1)
        except:
            return np.nan  # If convergence fails

    def compute_roi(self, sim: Simulation) -> float:
        df = self.to_dataframe(sim)
        years = (sim.end - sim.start).days / 365.25
        # Initial investment: net cash out at start (from params if available, else approx min cumulative)
        initial = sim.params.get('closing_fees', 0.0) + sim.params.get('appraisal', 0.0) * sim.params.get('down_fraction', 0.0) - sim.params.get('heloc_draw', 0.0)
        if initial == 0: initial = -df['cumulative_cash'].min()  # Fallback
        ending_net = df['net_worth'][-1]
        return ((ending_net - initial) / initial / years) if initial > 0 else np.nan

    def compute_statistics(self) -> Dict:
        dfs = [self.to_dataframe(s) for s in self.sims]
        endings = {k: [df[k][-1] for df in dfs] for k in ['net_worth', 'cumulative_cash', 'property_value', 'total_loans']}
        irrs = [self.compute_irr(s) for s in self.sims]
        rois = [self.compute_roi(s) for s in self.sims]
        breakevens = [ (df[df['net_worth'] > 0].index.min() - df.index.min()).days / 365.25 if any(df['net_worth'] > 0) else np.nan for df in dfs ]
        stats = {
            'net_worth_mean': np.mean(endings['net_worth']), 'net_worth_std': np.std(endings['net_worth']),
            'net_worth_var_5pct': np.percentile(endings['net_worth'], 5),  # Value at Risk proxy (worst 5%)
            'cash_mean': np.mean(endings['cumulative_cash']), 'cash_std': np.std(endings['cumulative_cash']),
            'prop_mean': np.mean(endings['property_value']), 'prop_std': np.std(endings['property_value']),
            'irr_mean': np.nanmean(irrs), 'irr_std': np.nanstd(irrs),
            'roi_mean': np.nanmean(rois), 'roi_std': np.nanstd(rois),
            'breakeven_years_mean': np.nanmean(breakevens),
            'prob_positive_net_worth': np.mean([e > 0 for e in endings['net_worth']])
        }
        return stats

    def plot_cumulative_cash_flows(self, title: str = "Cumulative Cash Flows"):
        def get_color(fraction):
            if fraction <= 0.5:
                val = fraction / 0.5
                r = 1
                g = val
                b = val
            else:
                val = (fraction - 0.5) / 0.5
                r = 1 - val
                g = 1 - val
                b = 1
            return (r, g, b)

        dfs = [self.to_dataframe(sim) for sim in self.sims]
        endings_net = [df['cumulative_cash'].iloc[-1] + df['property_value'].iloc[-1] for df in dfs]
        sorted_indices = np.argsort(endings_net)
        positions = np.linspace(0, len(self.sims) - 1, 11, dtype=int)
        selected_indices = sorted_indices[positions]

        min_x = min(df.index.min() for df in dfs)
        max_x = max(df.index.max() for df in dfs)

        plt.figure(figsize=(12, 6))
        for df in dfs:
            plt.plot(df.index, df['cumulative_cash'], color='black', alpha=0.1)

        for i in reversed(range(11)):
            idx = selected_indices[i]
            percentile = i * 10
            label = f"{percentile}th percentile ({self.sims[idx].name})"
            color = get_color(i / 10.0)
            plt.plot(dfs[idx].index, dfs[idx]['cumulative_cash'], color=color, label=label)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative Cash")
        plt.xlim(min_x, max_x)
        plt.margins(x=0)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_property_values(self, title: str = "Property Values Over Time"):
        def get_color(fraction):
            if fraction <= 0.5:
                val = fraction / 0.5
                r = 1
                g = val
                b = val
            else:
                val = (fraction - 0.5) / 0.5
                r = 1 - val
                g = 1 - val
                b = 1
            return (r, g, b)

        dfs = [self.to_dataframe(sim) for sim in self.sims]
        endings_net = [df['cumulative_cash'].iloc[-1] + df['property_value'].iloc[-1] for df in dfs]
        sorted_indices = np.argsort(endings_net)
        positions = np.linspace(0, len(self.sims) - 1, 11, dtype=int)
        selected_indices = sorted_indices[positions]

        min_x = min(df.index.min() for df in dfs)
        max_x = max(df.index.max() for df in dfs)

        plt.figure(figsize=(12, 6))
        for df in dfs:
            plt.plot(df.index, df['property_value'], color='black', alpha=0.1)

        for i in reversed(range(11)):
            idx = selected_indices[i]
            percentile = i * 10
            label = f"{percentile}th percentile ({self.sims[idx].name})"
            color = get_color(i / 10.0)
            plt.plot(dfs[idx].index, dfs[idx]['property_value'], color=color, label=label)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Property Value")
        plt.xlim(min_x, max_x)
        plt.margins(x=0)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_net_worth(self, title: str = "Net Worth Over Time Percentiles"):
        def get_color(fraction):
            if fraction <= 0.5:
                val = fraction / 0.5
                r = 1
                g = val
                b = val
            else:
                val = (fraction - 0.5) / 0.5
                r = 1 - val
                g = 1 - val
                b = 1
            return (r, g, b)

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

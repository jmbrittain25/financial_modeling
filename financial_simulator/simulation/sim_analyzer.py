from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .sim import Simulation


class SimulationAnalyzer:
    def __init__(self, sims: List[Simulation]):
        self.sims = sims

    def to_dataframe(self, sim: Simulation) -> pd.DataFrame:
        data = [{'time': e.time, 'value': e.value} for e in sim.events]
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        daily_cash = df['value'].groupby(df.index).sum()
        cumulative = daily_cash.cumsum()
        # Add property value from state
        prop_df = pd.DataFrame.from_dict(sim.state_history, orient='index')['property_value']
        prop_df.index = pd.to_datetime(prop_df.index)
        return pd.DataFrame({'cash_flow': daily_cash, 'cumulative': cumulative, 'property_value': prop_df})

    def plot_cumulative_cash_flows(self, title: str = "Cumulative Cash Flows"):
        plt.figure(figsize=(12, 6))
        for sim in self.sims:
            df = self.to_dataframe(sim)
            plt.plot(df.index, df['cumulative'], label=sim.name, alpha=0.7)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative Cash")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_property_values(self, title: str = "Property Values Over Time"):
        plt.figure(figsize=(12, 6))
        for sim in self.sims:
            df = self.to_dataframe(sim)
            plt.plot(df.index, df['property_value'], label=sim.name, alpha=0.7)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Property Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_histogram_end_values(self, title: str = "Distribution of Ending Cumulative Cash"):
        endings = [self.to_dataframe(sim)['cumulative'].iloc[-1] for sim in self.sims]
        plt.figure(figsize=(10, 5))
        plt.hist(endings, bins=20, edgecolor='black')
        plt.title(title)
        plt.xlabel("Ending Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def compute_statistics(self) -> Dict:
        endings_cash = [self.to_dataframe(sim)['cumulative'].iloc[-1] for sim in self.sims]
        endings_prop = [self.to_dataframe(sim)['property_value'].iloc[-1] for sim in self.sims]
        return {
            'cash_mean': np.mean(endings_cash),
            'cash_std': np.std(endings_cash),
            'prop_mean': np.mean(endings_prop),
            'prop_std': np.std(endings_prop),
            'net_worth_mean': np.mean([c + p for c, p in zip(endings_cash, endings_prop)])
        }

    def compare_simulations(self, sim1: Simulation, sim2: Simulation, title: str = "Comparison of Cumulative Cash Flows"):
        plt.figure(figsize=(12, 6))
        df1 = self.to_dataframe(sim1)
        df2 = self.to_dataframe(sim2)
        plt.plot(df1.index, df1['cumulative'], label=sim1.name)
        plt.plot(df2.index, df2['cumulative'], label=sim2.name)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Cumulative Cash")
        plt.legend()
        plt.grid(True)
        plt.show()

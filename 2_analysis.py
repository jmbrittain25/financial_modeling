import os
import financial_simulator as fs


if __name__ == "__main__":

    output_dir = os.path.join("output", "20251230_test")

    print(f"Loading simulations from '{output_dir}'.")

    analyzer = fs.SimulationAnalyzer.from_directory(output_dir)
    print(analyzer.compute_statistics())
    analyzer.plot_cumulative_cash_flows(title="Cumulative Cash Flows")
    analyzer.plot_property_values(title="Property Values Over Time")
    analyzer.plot_histogram_end_values(title="Distribution of Ending Net Worth")

    print("Analysis complete.")

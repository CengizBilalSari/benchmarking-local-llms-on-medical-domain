"""
Script 07: Summarize Benchmark Results.
Reads a results CSV and prints key performance metrics.
"""

import os
import sys
import pandas as pd

def summarize_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    print(f"\n{'='*50}")
    print(f"📊 REPORT: {os.path.basename(filepath)}")
    print(f"{'='*50}")

    df = pd.read_csv(filepath)
    
    # Basic Metrics
    total_q = len(df)
    correct_q = df['is_correct'].sum()
    accuracy = (correct_q / total_q) * 100
    
    # Latency Metrics
    avg_latency = df['latency_sec'].mean()
    max_latency = df['latency_sec'].max()
    min_latency = df['latency_sec'].min()
    
    # Token Metrics
    total_tokens = df['output_tokens'].sum()
    avg_tokens = df['output_tokens'].mean()
    
    # Throughput (Tokens per second)
    # We calculate this per-row first to get an average speed
    df['tokens_per_sec'] = df['output_tokens'] / df['latency_sec']
    avg_speed = df['tokens_per_sec'].mean()

    print(f"✅ ACCURACY:          {accuracy:.2f}% ({correct_q}/{total_q})")
    print(f"⏱️  LATENCY (AVG):      {avg_latency:.2f} seconds")
    print(f"🚀 SPEED (AVG):        {avg_speed:.2f} tokens/sec")
    print(f"📝 TOKENS (AVG):       {avg_tokens:.1f} per answer")
    print(f"💎 TOKENS (TOTAL):     {total_tokens:,}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Use provided file
        summarize_file(sys.argv[1])
    else:
        # Try to find the latest file in data/results
        results_dir = "data/results"
        if os.path.exists(results_dir):
            files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=os.path.getctime)
                print(f"No file specified. Using latest results: {latest_file}")
                summarize_file(latest_file)
            else:
                print("No result files found in data/results/")
        else:
            print("Results directory not found.")

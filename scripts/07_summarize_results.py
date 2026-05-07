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
        return None

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
    df['tokens_per_sec'] = df['output_tokens'] / df['latency_sec']
    avg_speed = df['tokens_per_sec'].mean()

    return {
        "filename": os.path.basename(filepath),
        "accuracy": accuracy,
        "correct_q": correct_q,
        "total_q": total_q,
        "avg_latency": avg_latency,
        "avg_speed": avg_speed,
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens
    }

def print_report(s):
    print(f"\n{'='*50}")
    print(f"📊 REPORT: {s['filename']}")
    print(f"{'='*50}")
    print(f"✅ ACCURACY:          {s['accuracy']:.2f}% ({s['correct_q']}/{s['total_q']})")
    print(f"⏱️  LATENCY (AVG):      {s['avg_latency']:.2f} seconds")
    print(f"🚀 SPEED (AVG):        {s['avg_speed']:.2f} tokens/sec")
    print(f"📝 TOKENS (AVG):       {s['avg_tokens']:.1f} per answer")
    print(f"💎 TOKENS (TOTAL):     {s['total_tokens']:,}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    results_dir = "data/results"
    
    summaries = []
    
    if len(sys.argv) > 1:
        # Use provided file
        s = summarize_file(sys.argv[1])
        if s: summaries.append(s)
    else:
        # Summarize all files in data/results
        if os.path.exists(results_dir):
            files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]
            for f in files:
                s = summarize_file(f)
                if s: summaries.append(s)
        else:
            print(f"Results directory '{results_dir}' not found.")

    if summaries:
        # Sort by accuracy high to low
        summaries.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"🔍 Summarizing {len(summaries)} result files (Sorted by Accuracy)\n")
        for s in summaries:
            print_report(s)
    else:
        print("No results found to summarize.")

"""
Script 05: Evaluate Local LLMs with Reasoning, Latency, and Token Tracking.
"""

import os
import sys
import pandas as pd
import requests
import json
import time
from tqdm import tqdm
from datetime import datetime
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_env

# Configuration
env = load_env()
LOCAL_LLM_URL = env['local_llm_url']
MODEL_NAME = env['local_model_name']
INPUT_FILE = "data/medical_mcqs.csv"
OUTPUT_DIR = "data/results"

def ask_local_llm(question, options):
    prompt = f"""You are a senior medical expert. Solve the following Multiple Choice Question.
1. Think step-by-step to analyze the clinical case.
2. Rule out incorrect options.
3. Provide your final answer at the very end in this exact format: "Final Answer: [LETTER]"

Question: {question}
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Reasoning:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, 
        "max_tokens": 8192 
    }

    start_time = time.time()
    try:
        response = requests.post(f"{LOCAL_LLM_URL}/chat/completions", json=payload, timeout=300)
        response.raise_for_status()
        end_time = time.time()
        
        resp_json = response.json()
        raw_content = resp_json['choices'][0]['message']['content'].strip()
        
        # Token usage (most local LLM servers provide this in the OpenAI format)
        usage = resp_json.get('usage', {})
        output_tokens = usage.get('completion_tokens', 0)
        latency = end_time - start_time
        
        # Enhanced parsing
        bracket_match = re.findall(r'\[([A-D])\]', raw_content.upper())
        if bracket_match:
            parsed_letter = bracket_match[-1]
        else:
            text_match = re.findall(r'(?:ANSWER|FINAL ANSWER|CORRECT OPTION)(?:\s*(?:IS|:))\s*([A-D])', raw_content.upper())
            if text_match:
                parsed_letter = text_match[-1]
            else:
                end_snippet = raw_content[-50:].upper()
                last_letter = re.findall(r'\b([A-D])\b', end_snippet)
                parsed_letter = last_letter[-1] if last_letter else "Error"
        
        return parsed_letter, raw_content, latency, output_tokens
    except Exception as e:
        return "Error", f"Connection Error: {str(e)}", 0, 0

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Run 06_download_medqa.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    results = []
    
    print(f"Benchmarking REASONING model '{MODEL_NAME}' on {len(df)} questions...")

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        options = {
            'A': row['option_a'],
            'B': row['option_b'],
            'C': row['option_c'],
            'D': row['option_d']
        }
        
        llm_letter, llm_raw, latency, tokens = ask_local_llm(row['question'], options)
        correct_answer = str(row['correct_option']).upper().strip()
        
        is_correct = 1 if llm_letter == correct_answer else 0
        status = "✅ CORRECT" if is_correct else f"❌ WRONG (GT: {correct_answer})"
        
        # Real-time print with latency and tokens
        print(f"\n[{i+1}/{len(df)}] Q: {row['question'][:80]}...")
        print(f"      LLM Choice: {llm_letter} | {status}")
        print(f"      Latency: {latency:.2f}s | Tokens: {tokens}")
        
        if not is_correct:
            print(f"\n      --- FULL REASONING ---")
            print(llm_raw) 
            print(f"      ----------------------")

        results.append({
            'question': row['question'],
            'ground_truth': correct_answer,
            'llm_choice': llm_letter,
            'llm_full_response': llm_raw,
            'is_correct': is_correct,
            'latency_sec': latency,
            'output_tokens': tokens
        })

    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_full_data_{MODEL_NAME.replace('/', '_')}_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    results_df.to_csv(output_path, index=False)
    
    # Summary
    accuracy = results_df['is_correct'].mean() * 100
    avg_latency = results_df['latency_sec'].mean()
    total_tokens = results_df['output_tokens'].sum()
    
    print(f"\n{'='*40}")
    print(f"✅ Evaluation Complete!")
    print(f"   Results: {output_path}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Avg Latency: {avg_latency:.2f}s")
    print(f"   Total Output Tokens: {total_tokens}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()

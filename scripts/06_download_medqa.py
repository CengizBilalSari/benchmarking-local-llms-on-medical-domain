"""
Script 06: Download and format the real MedQA (USMLE) dataset.
Ensures correct_option is a LETTER (A, B, C, D).
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Downloading MedQA (USMLE 4-options) from Hugging Face...")
    
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    mcq_records = []

    print(f"Formatting {len(dataset)} questions...")
    for item in tqdm(dataset):
        opts = item['options']
        
        # If the answer is the full text, we must find the corresponding letter
        correct_letter = item['answer']
        if len(correct_letter) > 1:
            # Match text to letter
            for letter, text in opts.items():
                if text == correct_letter:
                    correct_letter = letter
                    break
        
        mcq_records.append({
            'question': item['question'],
            'option_a': opts.get('A', ''),
            'option_b': opts.get('B', ''),
            'option_c': opts.get('C', ''),
            'option_d': opts.get('D', ''),
            'correct_option': correct_letter.upper()
        })

    df = pd.DataFrame(mcq_records)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    output_path = os.path.join(data_dir, 'medical_mcqs.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully fixed and saved {len(df)} MedQA questions to {output_path}")

if __name__ == '__main__':
    main()

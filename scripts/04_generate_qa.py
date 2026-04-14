"""
Script 04: Generate QA pairs from extracted multi-hop paths using OpenAI API.

Reads data/extracted_paths.json and generates natural-language question-answer
pairs using GPT-4o-mini. Each QA pair is grounded in a specific knowledge graph
path, ensuring the answer is derivable from the graph structure.

Output: data/qa_pairs.csv with columns:
  question, answer, hop_count, vertices, edges
"""

import os
import sys
import json
import time
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_env, get_openai_client, get_data_dir

MODEL = "gpt-4o-mini"
BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 1.0  # seconds
MAX_RETRIES = 3


def build_path_description(path: dict) -> str:
    """Convert a path dict into a human-readable description for the LLM."""
    vertices = path['vertices']
    edges = path['edges']

    lines = []
    for i, vertex in enumerate(vertices):
        node_type = vertex.get('type', 'Unknown')
        node_name = vertex.get('name', 'Unknown')
        lines.append(f"  Node {i+1}: [{node_type}] {node_name}")

        if i < len(edges):
            edge = edges[i]
            rel = edge.get('relation', 'Unknown')
            display = edge.get('display_relation', rel)
            lines.append(f"    --[{display}]-->")

    return '\n'.join(lines)


def build_prompt(path: dict) -> str:
    """Build the LLM prompt for QA generation from a path."""
    path_desc = build_path_description(path)
    hop_count = path['hop_count']

    return f"""You are a biomedical expert creating a medical knowledge benchmark dataset.

Given the following knowledge graph path ({hop_count} hops), generate a natural, clinically-relevant question and its answer.

Knowledge Graph Path:
{path_desc}

Requirements:
1. The question should be natural and sound like something a medical professional, researcher, or student would ask.
2. The answer MUST be derivable ONLY from the path information provided. Do not add external knowledge.
3. The answer should be concise (1-3 sentences).
4. The question should require reasoning across at least {min(hop_count, 3)} hops of the path — do NOT ask simple single-hop factual questions.
5. Frame the question so that the answer reveals the multi-step relationship chain.
6. Avoid yes/no questions. Ask "what", "which", "how", or "through what mechanism" questions.

Return your response in this EXACT JSON format (no markdown, no code fences):
{{"question": "your question here", "answer": "your answer here"}}"""


def generate_qa_for_path(client, path: dict, retries=MAX_RETRIES) -> dict:
    """Generate a QA pair for a single path using the OpenAI API."""
    prompt = build_prompt(path)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical expert. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Clean up potential markdown fences
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

            qa = json.loads(content)

            if 'question' not in qa or 'answer' not in qa:
                raise ValueError("Missing 'question' or 'answer' in response")

            return qa

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                'question': f"[PARSE_ERROR] Failed to parse LLM response for path",
                'answer': content if 'content' in dir() else "N/A"
            }

        except Exception as e:
            error_msg = str(e)
            if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                wait_time = 2 ** (attempt + 2)
                print(f"\n     Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                'question': f"[ERROR] {error_msg}",
                'answer': "N/A"
            }


def main():
    """Main QA generation pipeline."""
    print("=" * 60)
    print("QA Pair Generation from Multi-Hop Paths")
    print("=" * 60)

    # Load paths
    data_dir = get_data_dir()
    paths_file = os.path.join(data_dir, 'extracted_paths.json')

    if not os.path.exists(paths_file):
        print(f" {paths_file} not found. Run 03_extract_paths.py first.")
        sys.exit(1)

    with open(paths_file, 'r', encoding='utf-8') as f:
        paths = json.load(f)

    print(f"\n📂 Loaded {len(paths)} paths from {paths_file}")
    for hc in [3, 5, 7]:
        count = sum(1 for p in paths if p['hop_count'] == hc)
        print(f"   {hc}-hop: {count} paths")

    # Initialize OpenAI client
    env = load_env()
    client = get_openai_client(env)
    print(f"\n Using model: {MODEL}")

    # Generate QA pairs
    qa_records = []
    errors = 0

    print(f"\n Generating QA pairs ({len(paths)} total)...")

    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Batches"):
        batch = paths[i:i + BATCH_SIZE]

        for path in batch:
            qa = generate_qa_for_path(client, path)

            record = {
                'question': qa['question'],
                'answer': qa['answer'],
                'hop_count': path['hop_count'],
                'vertices': json.dumps(path['vertices'], ensure_ascii=False),
                'edges': json.dumps(path['edges'], ensure_ascii=False),
            }
            qa_records.append(record)

            if qa['question'].startswith('[ERROR]') or qa['question'].startswith('[PARSE_ERROR]'):
                errors += 1

        # Rate limiting delay between batches
        if i + BATCH_SIZE < len(paths):
            time.sleep(DELAY_BETWEEN_BATCHES)

    # Create DataFrame and save
    df = pd.DataFrame(qa_records)

    output_path = os.path.join(data_dir, 'qa_pairs.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"✅ QA Generation Complete!")
    print(f"   Total QA pairs: {len(df)}")
    print(f"   Errors: {errors}")
    print(f"   Success rate: {(len(df) - errors) / len(df) * 100:.1f}%")
    print(f"\n   Distribution by hop count:")
    for hc in sorted(df['hop_count'].unique()):
        count = len(df[df['hop_count'] == hc])
        print(f"     {hc}-hop: {count} pairs")

    print(f"\n📁 Saved to: {output_path}")

    # Print some examples
    print(f"\n{'─' * 60}")
    print(" Sample QA Pairs:")
    for hc in sorted(df['hop_count'].unique()):
        subset = df[df['hop_count'] == hc]
        if len(subset) > 0:
            row = subset.iloc[0]
            if not row['question'].startswith('['):
                print(f"\n   [{hc}-hop]")
                print(f"   Q: {row['question']}")
                print(f"   A: {row['answer']}")


if __name__ == '__main__':
    main()

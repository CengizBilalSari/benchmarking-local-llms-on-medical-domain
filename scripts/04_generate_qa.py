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

    # Detect which entity types appear so we can give domain-specific guidance
    node_types = set(v.get('type', '') for v in path['vertices'])
    edge_types = set(e.get('relation', '') for e in path['edges'])

    domain_hints = []
    if 'Drug' in node_types and 'Disease' in node_types:
        domain_hints.append("- Frame the question around pharmacology, treatment options, or therapeutic relationships (e.g. 'Why might a clinician prescribe X for a patient with Y?').")
    if 'GeneProtein' in node_types and 'Disease' in node_types:
        domain_hints.append("- Frame the question around genetic basis, molecular mechanisms, or pathophysiology (e.g. 'What molecular mechanism links disease X to phenotype Y?').")
    if 'EffectPhenotype' in node_types:
        domain_hints.append("- Frame the question around clinical presentation, symptoms, or adverse effects (e.g. 'Why might a patient on drug X develop symptom Y?').")
    if 'Pathway' in node_types:
        domain_hints.append("- Frame the question around biochemical pathways or metabolic processes (e.g. 'Through which metabolic pathway does protein X contribute to condition Y?').")
    if 'Anatomy' in node_types:
        domain_hints.append("- Frame the question around anatomical localization or tissue-specific expression (e.g. 'In which tissue is protein X expressed, and how does that relate to disease Y?').")
    if 'DRUG_EFFECT' in edge_types:
        domain_hints.append("- Focus on side effects and adverse drug reactions.")
    if 'OFF_LABEL_USE' in edge_types:
        domain_hints.append("- Mention off-label therapeutic uses where relevant.")
    if 'CONTRAINDICATION' in edge_types:
        domain_hints.append("- Focus on why a drug may be contraindicated for certain conditions.")
    if not domain_hints:
        domain_hints.append("- Frame the question around disease classification, differential diagnosis, or nosology (e.g. 'How are these conditions related in the disease taxonomy?').")

    domain_section = "\n".join(domain_hints)

    return f"""You are a senior physician and biomedical researcher writing exam questions for a medical licensing board.

Given the following knowledge graph path ({hop_count} hops), generate a realistic clinical or biomedical question and a clear, concise answer.

Knowledge Graph Path:
{path_desc}

Domain-specific guidance:
{domain_section}

STRICT REQUIREMENTS:
1. Write the question as a REAL MEDICAL QUESTION — the kind found in USMLE, clinical pharmacology exams, or medical board reviews. Do NOT mention "knowledge graph", "path", "hops", "nodes", or "edges".
2. The question should test clinical reasoning, mechanistic understanding, or therapeutic knowledge — NOT graph traversal.
3. The answer MUST be grounded in the relationships shown in the path. Do not fabricate information beyond what the path provides.
4. The answer should be concise (1-3 sentences) and clinically informative.
5. The question must require multi-step reasoning that spans at least {min(hop_count, 3)} entities in the path.
6. Avoid yes/no questions. Use "What", "Which", "How", "Why", or "Through what mechanism" question stems.
7. Do NOT simply list the entities. Instead, ask about the clinical or biological SIGNIFICANCE of their relationships.

GOOD question examples:
- "Why might a patient with Bernard-Soulier syndrome present with thrombocytopenia, and what gene is implicated?"
- "Through what molecular pathway could mutations in ACO2 lead to infantile cerebellar-retinal degeneration?"
- "A patient is prescribed Bumetanide for hypertension. What renal complication should be monitored?"

BAD question examples (DO NOT generate these):
- "What entities connect Disease A to Drug B?" (graph-speak)
- "Trace the path from node 1 to node 4." (graph-speak)
- "What is the relationship between A and B?" (too vague)

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

    for i in tqdm(range(0, 20, BATCH_SIZE), desc="Batches"):
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

    output_path = os.path.join(data_dir, 'qa_pairs_new.csv')
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

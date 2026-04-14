"""
Script 03: Extract multi-hop paths from Neo4j.

Extracts random multi-hop paths (3, 5, 7 hops) from the PrimeKG graph.
Uses a fast streaming approach: picks random start nodes and finds paths
iteratively, updating a real-time progress bar until the target is met.
"""

import os
import sys
import json
import random
import time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_env, get_neo4j_driver, get_data_dir

# Hop counts: 3 (baseline), 5 (moderate), 7 (hard)
HOP_COUNTS = [3, 5, 7]

# Clinical node label filter used in Cypher WHERE clause
CLINICAL_FILTER = "(n:Disease OR n:Drug OR n:GeneProtein OR n:EffectPhenotype OR n:Anatomy OR n:Pathway)"


def get_start_nodes(driver):
    """Gets all Disease and Drug nodes to use as starting points."""
    print(" Fetching candidate start nodes (Disease & Drug)...")
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n:Disease OR n:Drug RETURN n.node_index AS idx"
        )
        nodes = [row["idx"] for row in result]
    print(f"   Found {len(nodes):,} candidate start nodes.")
    return nodes


def extract_paths_for_hop(driver, hop_count, start_nodes, target_count):
    """
    Extract paths by picking random start nodes and query a small number of paths (LIMIT 10).
    Because we don't use 'ORDER BY rand()' in Cypher, Neo4j returns paths instantly
    as soon as it finds them via DFS, making this incredibly fast.
    """
    random.shuffle(start_nodes)
    all_paths = []
    seen_paths = set()
    
    # Excluded DRUG_DRUG and PROTEIN_PROTEIN because they are too densely connected.
    query = f"""
    MATCH path = (start)-[*{hop_count}]-(end)
    WHERE start.node_index = $start_idx
      AND start <> end
      AND ALL(n IN nodes(path) WHERE {CLINICAL_FILTER})
      AND NONE(r IN relationships(path) WHERE type(r) = 'DRUG_DRUG' OR type(r) = 'PROTEIN_PROTEIN')
    RETURN 
        [n IN nodes(path) | {{
            type: labels(n)[0], 
            name: n.name, 
            node_index: n.node_index
        }}] AS vertices,
        [rel IN relationships(path) | {{
            relation: type(rel), 
            display_relation: rel.display_relation
        }}] AS edges
    LIMIT 50
    """
    
    with driver.session() as session:
        with tqdm(total=target_count, desc=f"   {hop_count}-hop", unit=" paths") as pbar:
            for start_idx in start_nodes:
                if len(all_paths) >= target_count:
                    break
                    
                try:
                    # Execute query for this specific start node
                    result = session.run(query, start_idx=start_idx)
                    
                    for record in result:
                        vertices = record['vertices']
                        
                        # FAST CYCLE DETECTION IN PYTHON
                        # If a node appears twice in the path, it's a cycle. Skip it.
                        node_names = [v['name'] for v in vertices]
                        if len(set(node_names)) < len(node_names):
                            continue
                            
                        path_dict = {
                            'hop_count': hop_count,
                            'vertices': vertices,
                            'edges': record['edges']
                        }
                        
                        # Deduplicate across all found paths
                        key = tuple(node_names)
                        if key not in seen_paths:
                            seen_paths.add(key)
                            all_paths.append(path_dict)
                            pbar.update(1)
                            
                            # Break out of the inner loop so we ONLY take 1 path per start node!
                            # This guarantees maximum diversity (no duplicate start nodes).
                            break
                                
                except Exception as e:
                    # Ignore specific Neo4j memory errors for highly dense nodes
                    continue

    return all_paths[:target_count]


def main():
    """Main path extraction pipeline."""
    print("=" * 60)
    print("Multi-Hop Path Extraction from PrimeKG")
    print("Hop counts: 3 (baseline), 5 (moderate), 7 (hard)")
    print("=" * 60)

    env = load_env()
    driver = get_neo4j_driver(env)

    # Verify Neo4j connection
    try:
        with driver.session() as session:
            cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()['cnt']
            print(f"\n Neo4j contains {cnt:,} nodes")
            if cnt == 0:
                print(" No nodes found. Run 02_import_to_neo4j.py first.")
                sys.exit(1)
    except Exception as e:
        print(f" Cannot connect to Neo4j: {e}")
        print("   Make sure Neo4j is running: docker compose up -d")
        sys.exit(1)

    # Pre-fetch all possible start nodes
    start_nodes = get_start_nodes(driver)

    all_paths = []
    targets = {3: 334, 5: 333, 7: 333}  # total = 1000

    for hop_count in HOP_COUNTS:
        print(f"\n{'─' * 60}")
        target = targets[hop_count]
        
        start_time = time.time()
        sampled = extract_paths_for_hop(driver, hop_count, start_nodes, target)
        elapsed = time.time() - start_time
        
        all_paths.extend(sampled)
        print(f"    Finished {hop_count}-hop in {elapsed:.1f}s")

        # Print an example
        if sampled:
            example = sampled[0]
            node_chain = " → ".join(f"[{v['type']}] {v['name']}" for v in example['vertices'])
            print(f"   Example: {node_chain}")

    # Save to JSON
    data_dir = get_data_dir()
    output_path = os.path.join(data_dir, 'extracted_paths.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_paths, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f" Extracted {len(all_paths)} total paths")
    for hc in HOP_COUNTS:
        count = sum(1 for p in all_paths if p['hop_count'] == hc)
        print(f"   {hc}-hop: {count} paths")
    print(f"\n📁 Saved to: {output_path}")

    driver.close()

if __name__ == '__main__':
    main()

"""
Script 02: Import PrimeKG CSV into Neo4j.

Reads data/kg.csv and creates:
  - Nodes with labels based on their type (Disease, Drug, GeneProtein, etc.)
  - Edges with relationship types based on the relation column
  - Indexes for fast lookups

All ~4M edges and ~130K nodes are imported for full graph visualization.
"""

import os
import sys
import time
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    load_env, get_neo4j_driver, get_data_dir,
    sanitize_label, sanitize_rel_type
)

BATCH_SIZE = 5000


def wait_for_neo4j(driver, max_retries=30, delay=2):
    """Wait for Neo4j to be ready."""
    print("⏳ Waiting for Neo4j to be ready...")
    for i in range(max_retries):
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            print("✅ Neo4j is ready!")
            return True
        except Exception:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                print(" Neo4j is not responding. Make sure Docker is running:")
                print("   docker compose up -d")
                return False
    return False


def clear_database(driver):
    """Clear all existing data in the database."""
    print("  Clearing existing database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("   Done.")


def create_indexes(driver, node_types):
    """Create indexes for fast node lookups."""
    print("Creating indexes...")
    with driver.session() as session:
        for node_type in node_types:
            label = sanitize_label(node_type)
            try:
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.node_index)"
                )
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
                )
                print(f"    Index on :{label}(node_index, name)")
            except Exception as e:
                print(f"     Index on :{label}: {e}")
    print("   Done.")


def collect_unique_nodes(df):
    """Extract all unique nodes from both x and y columns."""
    print(" Collecting unique nodes...")

    x_nodes = df[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']].copy()
    x_nodes.columns = ['node_index', 'node_id', 'node_type', 'name', 'source']

    y_nodes = df[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']].copy()
    y_nodes.columns = ['node_index', 'node_id', 'node_type', 'name', 'source']

    all_nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates(subset=['node_index'])

    print(f"   Found {len(all_nodes):,} unique nodes")
    for node_type, count in all_nodes['node_type'].value_counts().items():
        print(f"     - {node_type}: {count:,}")

    return all_nodes


def import_nodes(driver, nodes_df):
    """Import nodes into Neo4j in batches, grouped by type."""
    print("\n📦 Importing nodes...")

    total_imported = 0
    grouped = nodes_df.groupby('node_type')

    for node_type, group in grouped:
        label = sanitize_label(node_type)
        records = group.to_dict('records')
        n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"   Importing {len(records):,} :{label} nodes...")

        for i in tqdm(range(n_batches), desc=f"   {label}", leave=False):
            batch = records[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            query = f"""
            UNWIND $batch AS row
            MERGE (n:{label} {{node_index: row.node_index}})
            ON CREATE SET
                n.node_id = row.node_id,
                n.name = row.name,
                n.source = row.source,
                n.node_type = row.node_type
            """

            with driver.session() as session:
                session.run(query, batch=batch)

            total_imported += len(batch)

    print(f"    Imported {total_imported:,} nodes total")


def import_edges(driver, df):
    """Import edges into Neo4j in batches, grouped by relation type."""
    print("\n🔗 Importing edges...")

    total_imported = 0
    grouped = df.groupby('relation')

    for relation, group in grouped:
        rel_type = sanitize_rel_type(relation)
        display = group['display_relation'].iloc[0]

        # We need to know the source and target labels for efficient MATCH
        records = group[['x_index', 'x_type', 'y_index', 'y_type', 'display_relation']].to_dict('records')
        n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"   Importing {len(records):,} [{rel_type}] edges ({display})...")

        # Group by source_type -> target_type for label-specific MATCH
        sub_grouped = group.groupby(['x_type', 'y_type'])

        for (x_type, y_type), sub_group in sub_grouped:
            x_label = sanitize_label(x_type)
            y_label = sanitize_label(y_type)

            sub_records = sub_group[['x_index', 'y_index', 'display_relation']].to_dict('records')
            sub_batches = (len(sub_records) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in tqdm(range(sub_batches), desc=f"   {x_label}->{y_label}", leave=False):
                batch = sub_records[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

                query = f"""
                UNWIND $batch AS row
                MATCH (a:{x_label} {{node_index: row.x_index}})
                MATCH (b:{y_label} {{node_index: row.y_index}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.display_relation = row.display_relation
                """

                with driver.session() as session:
                    session.run(query, batch=batch)

                total_imported += len(batch)

    print(f"   Imported {total_imported:,} edges total")


def verify_import(driver):
    """Verify the import by running count queries."""
    print("\n Verifying import...")
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()['cnt']
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()['cnt']

        print(f"   Nodes in Neo4j: {node_count:,}")
        print(f"   Edges in Neo4j: {edge_count:,}")

        # Count by label
        labels = session.run("CALL db.labels() YIELD label RETURN label").data()
        print("\n   Nodes by label:")
        for record in labels:
            label = record['label']
            cnt = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS cnt"
            ).single()['cnt']
            print(f"     - :{label}: {cnt:,}")

        # Count by relationship type
        rel_types = session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        ).data()
        print(f"\n   Relationship types: {len(rel_types)}")
        for record in rel_types:
            rt = record['relationshipType']
            cnt = session.run(
                f"MATCH ()-[r:{rt}]->() RETURN count(r) AS cnt"
            ).single()['cnt']
            print(f"     - [{rt}]: {cnt:,}")


def main():
    """Main import pipeline."""
    print("=" * 60)
    print("PrimeKG → Neo4j Import Pipeline")
    print("=" * 60)

    # Load CSV
    data_dir = get_data_dir()
    csv_path = os.path.join(data_dir, 'kg.csv')

    if not os.path.exists(csv_path):
        print(f" {csv_path} not found. Run 01_download_primekg.py first.")
        sys.exit(1)

    print(f"\n📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df):,} edges")

    # Connect to Neo4j
    env = load_env()
    driver = get_neo4j_driver(env)

    if not wait_for_neo4j(driver):
        sys.exit(1)

    # Clear and rebuild
    clear_database(driver)

    # Get unique node types and create indexes
    node_types = set(df['x_type'].unique()) | set(df['y_type'].unique())
    create_indexes(driver, node_types)

    # Collect and import nodes
    nodes_df = collect_unique_nodes(df)
    import_nodes(driver, nodes_df)

    # Import edges
    import_edges(driver, df)

    # Verify
    verify_import(driver)

    driver.close()
    print("\n Import complete! Visit http://localhost:7474 to explore the graph.")


if __name__ == '__main__':
    main()

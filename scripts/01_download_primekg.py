"""
Script 01: Download PrimeKG CSV from Harvard Dataverse.

Downloads the kg.csv file (~300MB) and saves it to data/kg.csv.
Prints basic statistics about the dataset after download.
"""

import os
import sys
import requests
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import get_data_dir

PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
FILENAME = "kg.csv"


def download_primekg(data_dir: str, force: bool = False):
    """Download PrimeKG CSV from Harvard Dataverse with progress bar."""
    filepath = os.path.join(data_dir, FILENAME)

    if os.path.exists(filepath) and not force:
        print(f" {FILENAME} already exists at {filepath}")
        print(" Use --force flag to re-download.")
        return filepath

    print(f"Downloading PrimeKG from Harvard Dataverse...")
    print(f"URL: {PRIMEKG_URL}")
    print(f"Destination: {filepath}")

    response = requests.get(PRIMEKG_URL, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Download complete: {filepath}")
    return filepath


def print_stats(filepath: str):
    """Print basic statistics about the downloaded PrimeKG CSV."""
    print("\n PrimeKG Dataset Statistics:")
    print("-" * 50)

    # Read just enough to get stats without loading everything into memory
    print("Loading CSV")
    df = pd.read_csv(filepath)

    print(f"   Total edges (rows): {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    # Node type stats
    x_types = set(df['x_type'].unique())
    y_types = set(df['y_type'].unique())
    all_types = x_types | y_types
    print(f"\n   Node types ({len(all_types)}):")
    for t in sorted(all_types):
        # Count unique nodes of this type
        x_count = df[df['x_type'] == t]['x_index'].nunique()
        y_count = df[df['y_type'] == t]['y_index'].nunique()
        print(f"     - {t}: ~{x_count + y_count:,} nodes")

    # Relation stats
    relations = df['relation'].value_counts()
    print(f"\n   Relation types ({len(relations)}):")
    for rel, count in relations.items():
        display = df[df['relation'] == rel]['display_relation'].iloc[0]
        print(f"     - {rel} ({display}): {count:,} edges")

    # Unique nodes
    x_nodes = set(df['x_index'].unique())
    y_nodes = set(df['y_index'].unique())
    all_nodes = x_nodes | y_nodes
    print(f"\n   Total unique nodes: {len(all_nodes):,}")
    print(f"   Total edges: {len(df):,}")


if __name__ == '__main__':
    force = '--force' in sys.argv
    data_dir = get_data_dir()
    filepath = download_primekg(data_dir, force=force)
    print_stats(filepath)

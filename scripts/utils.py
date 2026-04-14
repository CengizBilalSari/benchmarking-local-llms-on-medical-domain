"""
Shared utilities for the PrimeKG benchmarking pipeline.
Handles environment loading, Neo4j connection, and label sanitization.
"""

import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI


def load_env():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
    return {
        'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD', 'primekg_benchmark'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }


def get_neo4j_driver(env=None):
    """Create and return a Neo4j driver instance."""
    if env is None:
        env = load_env()
    return GraphDatabase.driver(
        env['neo4j_uri'],
        auth=(env['neo4j_user'], env['neo4j_password'])
    )


def get_openai_client(env=None):
    """Create and return an OpenAI client instance."""
    if env is None:
        env = load_env()
    api_key = env['openai_api_key']
    if not api_key or api_key == 'sk-your-key-here':
        raise ValueError(
            "OpenAI API key not set. Please update OPENAI_API_KEY in .env file."
        )
    return OpenAI(api_key=api_key)


def sanitize_label(node_type: str) -> str:
    """
    Convert PrimeKG node type to a valid Neo4j label.
    
    Examples:
        'gene/protein' -> 'GeneProtein'
        'disease'      -> 'Disease'
        'drug'         -> 'Drug'
        'biological_process' -> 'BiologicalProcess'
    """
    # Replace non-alphanumeric with space, then title-case
    cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', node_type)
    return ''.join(word.capitalize() for word in cleaned.split())


def sanitize_rel_type(relation: str) -> str:
    """
    Convert PrimeKG relation to a valid Neo4j relationship type.
    
    Examples:
        'disease_protein'  -> 'DISEASE_PROTEIN'
        'drug-target'      -> 'DRUG_TARGET'
        'protein_protein'  -> 'PROTEIN_PROTEIN'
    """
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', relation)
    return cleaned.upper()


# Node types considered clinically relevant for QA extraction
CLINICAL_NODE_TYPES = {
    'disease',
    'drug',
    'gene/protein',
    'effect/phenotype',
    'anatomy',
    'pathway',
}

# Neo4j labels for clinical node types
CLINICAL_LABELS = [sanitize_label(t) for t in CLINICAL_NODE_TYPES]


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir():
    """Get the data directory, creating it if necessary."""
    data_dir = os.path.join(get_project_root(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

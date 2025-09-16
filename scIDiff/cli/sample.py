#!/usr/bin/env python3
"""
Sampling CLI for scIDiff
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from ..models import ScIDiffModel


def main():
    parser = argparse.ArgumentParser(description="Sample from scIDiff model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, default="./samples.csv", help="Output samples path")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--gene-dim", type=int, required=True, help="Number of genes")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--gene-names", type=str, help="Path to file with gene names (one per line)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    model = ScIDiffModel(gene_dim=args.gene_dim, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generate samples
    print(f"Generating {args.num_samples} samples")
    with torch.no_grad():
        samples = model.sample(batch_size=args.num_samples)
        samples = samples.cpu().numpy()
    
    # Load gene names if provided
    if args.gene_names:
        with open(args.gene_names, 'r') as f:
            gene_names = [line.strip() for line in f]
        if len(gene_names) != args.gene_dim:
            print(f"Warning: {len(gene_names)} gene names provided but model expects {args.gene_dim}")
            gene_names = [f"gene_{i}" for i in range(args.gene_dim)]
    else:
        gene_names = [f"gene_{i}" for i in range(args.gene_dim)]
    
    # Save samples
    df = pd.DataFrame(samples, columns=gene_names)
    df.to_csv(args.output, index=False)
    print(f"Samples saved to {args.output}")


if __name__ == "__main__":
    main()


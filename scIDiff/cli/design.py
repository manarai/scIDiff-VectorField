#!/usr/bin/env python3
"""
Inverse design CLI for scIDiff
"""

import argparse
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path

from ..models import ScIDiffModel
from ..sampling import InverseDesigner, PhenotypeTarget


def main():
    parser = argparse.ArgumentParser(description="Inverse design with scIDiff")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--targets", type=str, required=True, help="Path to JSON file with gene targets")
    parser.add_argument("--output", type=str, default="./designed_cells.csv", help="Output path")
    parser.add_argument("--gene-dim", type=int, required=True, help="Number of genes")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for design")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--gene-names", type=str, help="Path to file with gene names (one per line)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load targets
    with open(args.targets, 'r') as f:
        targets_dict = json.load(f)
    
    target = PhenotypeTarget(gene_targets=targets_dict)
    print(f"Loaded targets for {len(targets_dict)} genes")
    
    # Load gene names if provided
    if args.gene_names:
        with open(args.gene_names, 'r') as f:
            gene_names = [line.strip() for line in f]
        if len(gene_names) != args.gene_dim:
            print(f"Warning: {len(gene_names)} gene names provided but model expects {args.gene_dim}")
            gene_names = [f"gene_{i}" for i in range(args.gene_dim)]
    else:
        gene_names = [f"gene_{i}" for i in range(args.gene_dim)]
    
    # Load model
    print(f"Loading model from {args.model}")
    model = ScIDiffModel(gene_dim=args.gene_dim, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create designer
    designer = InverseDesigner(model, var_names=gene_names)
    
    # Design cells
    print(f"Designing cells with batch size {args.batch_size}")
    designed_cells = designer.design(target, batch_size=args.batch_size)
    designed_cells = designed_cells.cpu().numpy()
    
    # Save results
    df = pd.DataFrame(designed_cells, columns=gene_names)
    df.to_csv(args.output, index=False)
    print(f"Designed cells saved to {args.output}")


if __name__ == "__main__":
    main()


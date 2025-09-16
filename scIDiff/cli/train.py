#!/usr/bin/env python3
"""
Training CLI for scIDiff
"""

import argparse
import torch
from torch.utils.data import DataLoader
import scanpy as sc
import pandas as pd
from pathlib import Path

from ..models import ScIDiffModel
from ..training import ScIDiffTrainer
from ..data import SingleCellDataset


def main():
    parser = argparse.ArgumentParser(description="Train scIDiff model")
    parser.add_argument("--data", type=str, required=True, help="Path to h5ad file")
    parser.add_argument("--output", type=str, default="./scidiff_model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data}")
    adata = sc.read_h5ad(args.data)
    
    # Prepare data
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    dataset = SingleCellDataset(X, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    gene_dim = X.shape[1]
    model = ScIDiffModel(gene_dim=gene_dim, hidden_dim=args.hidden_dim)
    model = model.to(device)
    
    print(f"Model created with {gene_dim} genes, {args.hidden_dim} hidden dim")
    
    # Create trainer
    trainer = ScIDiffTrainer(model, dataloader, lr=args.lr)
    
    # Train
    print(f"Starting training for {args.epochs} epochs")
    trainer.train(num_epochs=args.epochs)
    
    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()


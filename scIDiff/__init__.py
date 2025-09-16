"""
scIDiff: Single-cell Inverse Diffusion

A deep generative framework for modeling, denoising, and inverse-designing 
single-cell gene expression profiles using score-based diffusion models with
Dynamo vector field integration and Optimal Transport guidance.
"""

__version__ = "0.2.0"
__author__ = "scIDiff Team"
__email__ = "contact@scidiff.org"

# Import main components for easy access
from .models import ScIDiffModel
from .training import ScIDiffTrainer
from .sampling import InverseDesigner, PhenotypeTarget
from .data import SingleCellDataset

__all__ = [
    "ScIDiffModel",
    "ScIDiffTrainer", 
    "InverseDesigner",
    "PhenotypeTarget",
    "SingleCellDataset",
]


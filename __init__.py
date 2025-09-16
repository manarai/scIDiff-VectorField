"""
scIDiff: Single-cell Inverse Diffusion

A deep generative framework for modeling, denoising, and inverse-designing 
single-cell gene expression profiles using score-based diffusion models.
"""

__version__ = "0.1.0"
__author__ = "scIDiff Team"
__email__ = "contact@scidiff.org"

# Import main components for easy access
from .models import ScIDiffModel
from .training import ScIDiffTrainer
from .sampling import InverseDesigner, PhenotypeTarget

__all__ = [
    "ScIDiffModel",
    "ScIDiffTrainer", 
    "InverseDesigner",
    "PhenotypeTarget",
]


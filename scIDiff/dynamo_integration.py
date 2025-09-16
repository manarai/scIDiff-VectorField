"""
Enhanced Dynamo integration for scIDiff with vector field velocyto
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
import warnings

try:
    import dynamo as dyn
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False
    warnings.warn("Dynamo not available. Install with: pip install dynamo-release")

try:
    import scvelo as scv
    SCVELO_AVAILABLE = True
except ImportError:
    SCVELO_AVAILABLE = False
    warnings.warn("scVelo not available. Install with: pip install scvelo")


class DynamoVectorField:
    """
    Wrapper for Dynamo vector field with scIDiff integration
    """
    
    def __init__(
        self,
        adata=None,
        basis: str = 'pca',
        n_pca_components: int = 50,
        method: str = 'SparseVFC'
    ):
        self.adata = adata
        self.basis = basis
        self.n_pca_components = n_pca_components
        self.method = method
        self.vf_dict = None
        self.is_fitted = False
        
        if not DYNAMO_AVAILABLE:
            raise ImportError("Dynamo is required for vector field analysis")
    
    def preprocess_data(self, adata):
        """Preprocess data for Dynamo analysis"""
        if not SCVELO_AVAILABLE:
            warnings.warn("scVelo not available, using basic preprocessing")
            return adata
        
        # Basic preprocessing with scVelo
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        
        return adata
    
    def fit_vector_field(
        self,
        adata=None,
        layer: str = 'X_spliced',
        **kwargs
    ):
        """
        Fit Dynamo vector field to the data
        """
        if adata is None:
            adata = self.adata
        
        if adata is None:
            raise ValueError("No data provided for vector field fitting")
        
        # Preprocess if needed
        if 'X_spliced' not in adata.layers:
            adata = self.preprocess_data(adata)
        
        # Dynamo preprocessing
        dyn.pp.recipe_monocle(adata)
        
        # Fit vector field
        dyn.tl.dynamics(adata, model='stochastic')
        dyn.tl.reduceDimension(adata, basis=self.basis, n_pca_components=self.n_pca_components)
        dyn.tl.cell_velocities(adata, basis=self.basis)
        
        # Vector field reconstruction
        dyn.vf.VectorField(adata, basis=self.basis, method=self.method, **kwargs)
        
        self.vf_dict = adata.uns[f'VecFld_{self.basis}']
        self.is_fitted = True
        self.adata = adata
        
        return self
    
    def predict_velocity(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity for given gene expression states
        """
        if not self.is_fitted:
            raise ValueError("Vector field not fitted. Call fit_vector_field first.")
        
        # Convert to numpy for Dynamo
        X_np = X.detach().cpu().numpy()
        
        # Use Dynamo's vector field function
        velocities = dyn.vf.vector_field_function(
            X_np, 
            self.vf_dict
        )
        
        # Convert back to torch
        return torch.tensor(velocities, dtype=X.dtype, device=X.device)
    
    def get_pytorch_surrogate(self, hidden_dim: int = 512) -> nn.Module:
        """
        Create a PyTorch surrogate model for the vector field
        """
        if not self.is_fitted:
            raise ValueError("Vector field not fitted. Call fit_vector_field first.")
        
        # Create surrogate network
        surrogate = VectorFieldSurrogate(
            input_dim=self.n_pca_components,
            hidden_dim=hidden_dim
        )
        
        # Generate training data from Dynamo
        n_samples = 10000
        X_train = np.random.randn(n_samples, self.n_pca_components)
        V_train = dyn.vf.vector_field_function(X_train, self.vf_dict)
        
        # Train surrogate
        surrogate.fit(X_train, V_train)
        
        return surrogate


class VectorFieldSurrogate(nn.Module):
    """
    Neural network surrogate for Dynamo vector field
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def fit(
        self,
        X_train: np.ndarray,
        V_train: np.ndarray,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int = 256
    ):
        """
        Fit surrogate to Dynamo vector field data
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        V_tensor = torch.tensor(V_train, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, V_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_v in dataloader:
                pred_v = self(batch_x)
                loss = nn.MSELoss()(pred_v, batch_v)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 100 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Surrogate training epoch {epoch}, loss: {avg_loss:.6f}")


class DynamoGuidedDiffusion(nn.Module):
    """
    Diffusion model with Dynamo vector field guidance
    """
    
    def __init__(
        self,
        score_network: nn.Module,
        vector_field: VectorFieldSurrogate,
        guidance_strength: float = 1.0
    ):
        super().__init__()
        self.score_network = score_network
        self.vector_field = vector_field
        self.guidance_strength = guidance_strength
    
    def guided_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute score with Dynamo vector field guidance
        """
        # Base score from diffusion model
        base_score = self.score_network(x, t, condition)
        
        # Vector field guidance
        vf_guidance = self.vector_field(x)
        
        # Combine with time-dependent weighting
        t_weight = torch.exp(-t).unsqueeze(-1)  # Stronger guidance at later times
        guided_score = base_score + self.guidance_strength * t_weight * vf_guidance
        
        return guided_score
    
    def sample_with_guidance(
        self,
        x_init: torch.Tensor,
        num_steps: int = 1000,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample with vector field guidance
        """
        x = x_init.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((x.size(0),), 1.0 - i * dt, device=x.device)
            
            # Get guided score
            score = self.guided_score(x, t, condition)
            
            # SDE step
            drift = -0.5 * x + score
            noise = torch.randn_like(x) * np.sqrt(dt)
            
            x = x + drift * dt + noise
        
        return x


class VelocytoIntegration:
    """
    Integration with RNA velocity analysis
    """
    
    def __init__(self):
        if not SCVELO_AVAILABLE:
            raise ImportError("scVelo is required for velocyto integration")
    
    def compute_rna_velocity(
        self,
        adata,
        mode: str = 'dynamical'
    ):
        """
        Compute RNA velocity using scVelo
        """
        # Preprocessing
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        
        # Velocity computation
        if mode == 'dynamical':
            scv.tl.recover_dynamics(adata)
            scv.tl.velocity(adata, mode='dynamical')
        else:
            scv.tl.velocity(adata, mode='stochastic')
        
        scv.tl.velocity_graph(adata)
        
        return adata
    
    def detect_perturbation_effects(
        self,
        adata_control,
        adata_perturbed,
        min_velocity_change: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect perturbation effects using velocity changes
        """
        # Compute velocities for both conditions
        adata_control = self.compute_rna_velocity(adata_control)
        adata_perturbed = self.compute_rna_velocity(adata_perturbed)
        
        # Compare velocity patterns
        vel_control = adata_control.layers['velocity']
        vel_perturbed = adata_perturbed.layers['velocity']
        
        # Compute velocity differences
        vel_diff = np.abs(vel_perturbed.mean(axis=0) - vel_control.mean(axis=0))
        
        # Identify significantly changed genes
        changed_genes = adata_control.var_names[vel_diff > min_velocity_change]
        
        return {
            'changed_genes': changed_genes.tolist(),
            'velocity_differences': vel_diff,
            'control_velocity': vel_control,
            'perturbed_velocity': vel_perturbed
        }


def create_dynamo_guided_model(
    gene_dim: int,
    adata=None,
    hidden_dim: int = 512,
    guidance_strength: float = 1.0
) -> DynamoGuidedDiffusion:
    """
    Factory function to create a Dynamo-guided diffusion model
    """
    from .models import ScIDiffModel
    
    # Create base diffusion model
    base_model = ScIDiffModel(gene_dim, hidden_dim)
    
    # Create and fit vector field
    if adata is not None:
        vf = DynamoVectorField(adata)
        vf.fit_vector_field()
        surrogate = vf.get_pytorch_surrogate(hidden_dim)
    else:
        # Create dummy surrogate for testing
        surrogate = VectorFieldSurrogate(gene_dim, hidden_dim)
    
    # Create guided model
    guided_model = DynamoGuidedDiffusion(
        base_model.score,
        surrogate,
        guidance_strength
    )
    
    return guided_model



import torch
from src.models.components.ae.set_scae import SetSCAE

def test_setscae_shapes():
    """Test SetSCAE model with different strategies."""
    
    strategies = ['graph', 'contrastive', 'masked', 'stack']
    n_input = 100
    n_hidden = 32
    n_latent = 10
    set_size = 5
    batch_size = 2
    
    x_set = torch.randn(batch_size, set_size, n_input)
    
    for strategy in strategies:
        print(f"\n--- Testing strategy: {strategy} ---")
        model = SetSCAE(
            strategy=strategy,
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            set_size=set_size
        )
        
        outputs = model(x_set)
        
        print(f"z_center shape: {outputs['z_center'].shape}")
        print(f"z_all shape: {outputs['z_all'].shape}")
        
        # Check latent calculation
        z_latent = model.get_latent(x_set)
        print(f"get_latent shape: {z_latent.shape}")

if __name__ == "__main__":
    test_setscae_shapes()

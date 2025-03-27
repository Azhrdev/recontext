#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Implicit Surface Representation module using NeRF-based techniques.
Implements a differentiable surface extraction method with Eikonal regularization.

Based on research from:
- Neural Radiance Fields (NeRF)
- Implicit Differentiable IsoSurface Extraction
- Neural Geometric Level of Detail

TODO: Add proper references section
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from recontext.utils.transforms import normalize_pointcloud

logger = logging.getLogger(__name__)

# Set to True to enable verbose output during training
DEBUG = False

class SineLayer(nn.Module):
    """Sine activation with frequency modulation adapted from SIREN."""
    
    def __init__(self, in_features, out_features, bias=True, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize weights following SIREN paper
        with torch.no_grad():
            if omega_0 > 0:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, 
                                         np.sqrt(6 / in_features) / omega_0)
            else:
                nn.init.xavier_normal_(self.linear.weight)
            
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ImplicitNetwork(nn.Module):
    """Neural network that represents signed distance function (SDF)."""
    
    def __init__(self, d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,), 
                 multires=6, bias=True, geometric_init=True):
        super().__init__()
        
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn = None
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        
        # Create layers
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                
            if l == 0:
                # First layer uses standard Linear
                layer = nn.Linear(dims[l], out_dim, bias=bias)
            else:
                # Hidden layers use SIREN activation
                layer = SineLayer(dims[l], out_dim, bias=bias, omega_0=30.0)
                
            if geometric_init and l == self.num_layers - 2:
                # Initialize last layer to represent an approximate unit sphere
                nn.init.normal_(layer.linear.weight, mean=0.0, std=0.1)
                nn.init.constant_(layer.linear.bias, 0.0)
            
            layer_name = f'lin{l+1}'
            setattr(self, layer_name, layer)
        
        # Activation functions
        self.activation = nn.Softplus(beta=100)
    
    def forward(self, inputs):
        x = inputs
        
        if self.embed_fn is not None:
            x = self.embed_fn(x)
        
        # Forward pass through layers
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, f'lin{l+1}')
            
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1)
            
            if l == 0:
                # First layer with ReLU
                x = F.relu(lin(x))
            elif l == self.num_layers - 2:
                # Last layer with tanh to bound output
                x = lin(x)
            else:
                # Hidden layers use SIREN activation
                x = lin(x)
        
        return x
    
    def gradient(self, x):
        """Compute gradient of SDF with respect to input."""
        x.requires_grad_(True)
        y = self.forward(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients


def get_embedder(multires, input_dims=3):
    """Positional encoding embedding function."""
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = PositionalEncoding(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo(x)
    return embed, embedder_obj.out_dim


class PositionalEncoding(nn.Module):
    """Positional encoding module that maps coordinates to higher dimensions."""
    
    def __init__(self, include_input=True, input_dims=3, max_freq_log2=9, 
                num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        super().__init__()
        
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dims
            
        self.out_dim += self.input_dims * len(self.periodic_fns) * self.num_freqs
        
        # Create frequency bands
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq_log2, steps=self.num_freqs)
            
        self.register_buffer('freq_bands', freq_bands, persistent=True)
            
    def forward(self, inputs):
        out = []
        
        if self.include_input:
            out.append(inputs)
            
        # Apply positional encoding
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                out.append(p_fn(freq * inputs))
                
        return torch.cat(out, dim=-1)


class NeuralImplicitTrainer:
    """Trainer for neural implicit surface representation."""
    
    def __init__(self, pointcloud, config=None):
        """Initialize trainer with point cloud data.
        
        Args:
            pointcloud: Open3D point cloud containing points and normals
            config: Optional configuration dictionary
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Default configuration
        self.config = {
            'learning_rate': 1e-4,
            'batch_size': 5000,
            'num_iterations': 10000,
            'eikonal_weight': 0.1,
            'normal_weight': 1.0,
            'points_sigma': 0.01,
            'multires': 6,
            'hidden_dim': 256,
            'num_layers': 8,
            'marching_cubes_resolution': 128,
            'bounding_box_padding': 0.1,
        }
        
        # Update with provided config
        if config is not None:
            self.config.update(config)
        
        # Process pointcloud data
        self.points = np.asarray(pointcloud.points)
        self.normals = np.asarray(pointcloud.normals)
        
        # Normalize pointcloud to fit in unit cube
        self.points, self.center, self.scale = normalize_pointcloud(self.points)
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        
        # Create network
        self.network = ImplicitNetwork(
            d_in=3,
            d_out=1,
            d_hidden=self.config['hidden_dim'],
            n_layers=self.config['num_layers'],
            multires=self.config['multires']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                         lr=self.config['learning_rate'])
        
        # Prepare data for training
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for training."""
        # Convert to torch tensors
        self.points_tensor = torch.from_numpy(self.points).float().to(self.device)
        self.normals_tensor = torch.from_numpy(self.normals).float().to(self.device)
        
        # Generate off-surface points
        noise = torch.randn_like(self.points_tensor) * self.config['points_sigma']
        self.off_points = self.points_tensor + noise
        
        # Number of points
        self.num_points = self.points_tensor.shape[0]
        logger.info(f"Training with {self.num_points} points")
    
    def train(self):
        """Train the neural implicit network."""
        logger.info("Starting neural implicit surface training...")
        
        # Training loop
        progress_bar = tqdm(range(self.config['num_iterations']), desc="Training")
        for i in progress_bar:
            loss = self._train_step()
            
            if i % 100 == 0:
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        logger.info("Training complete.")
    
    def _train_step(self):
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        # Sample batch
        idx = torch.randint(0, self.num_points, (self.config['batch_size'],))
        points_batch = self.points_tensor[idx]
        normals_batch = self.normals_tensor[idx]
        off_points_batch = self.off_points[idx]
        
        # Forward pass
        sdf_pred = self.network(points_batch)
        
        # SDF should be zero at surface points
        sdf_loss = torch.mean(torch.abs(sdf_pred))
        
        # Compute gradients (normals)
        gradients = self.network.gradient(points_batch)
        normal_loss = torch.mean(torch.abs(gradients - normals_batch))
        
        # Eikonal term (gradient should have unit norm)
        off_gradients = self.network.gradient(off_points_batch)
        eikonal_loss = torch.mean((torch.norm(off_gradients, dim=-1) - 1)**2)
        
        # Total loss
        loss = (sdf_loss + 
                self.config['normal_weight'] * normal_loss + 
                self.config['eikonal_weight'] * eikonal_loss)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def extract_mesh(self):
        """Extract mesh using marching cubes algorithm."""
        logger.info("Extracting mesh using marching cubes...")
        
        # Create grid
        resolution = self.config['marching_cubes_resolution']
        padding = self.config['bounding_box_padding']
        
        # Create grid in [-1-padding, 1+padding]^3
        min_bound = -1.0 - padding
        max_bound = 1.0 + padding
        
        xs = torch.linspace(min_bound, max_bound, resolution)
        ys = torch.linspace(min_bound, max_bound, resolution)
        zs = torch.linspace(min_bound, max_bound, resolution)
        
        # Create meshgrid
        z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
        points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1).to(self.device)
        
        # Evaluate SDF
        chunk_size = 32768  # Process in chunks to avoid OOM
        sdf_values = []
        
        with torch.no_grad():
            for i in range(0, points.shape[0], chunk_size):
                chunk_points = points[i:i+chunk_size]
                chunk_sdf = self.network(chunk_points).cpu().numpy()
                sdf_values.append(chunk_sdf)
        
        sdf_values = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
        
        # Run marching cubes
        try:
            import skimage.measure
            vertices, faces, normals, _ = skimage.measure.marching_cubes(
                sdf_values, level=0, spacing=(
                    (max_bound - min_bound) / (resolution - 1),
                    (max_bound - min_bound) / (resolution - 1),
                    (max_bound - min_bound) / (resolution - 1)
                )
            )
            
            # Rescale vertices to original coordinate system
            vertices = vertices + min_bound
            vertices = vertices * self.scale + self.center
            
            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            
            # Fix orientation
            mesh.compute_triangle_normals()
            
            logger.info(f"Mesh extracted with {len(mesh.triangles)} triangles")
            return mesh
            
        except Exception as e:
            logger.error(f"Error during marching cubes: {e}")
            return None


def optimize_neural_surface(pointcloud, images, cameras, config=None):
    """Optimize neural implicit surface representation.
    
    Args:
        pointcloud: Open3D point cloud
        images: List of images (for texture)
        cameras: Camera parameters
        config: Optional configuration
        
    Returns:
        Open3D mesh
    """
    # Ensure pointcloud has normals
    if not pointcloud.has_normals():
        logger.info("Computing point cloud normals")
        pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pointcloud.orient_normals_consistent_tangent_plane(100)
    
    # Create trainer
    trainer = NeuralImplicitTrainer(pointcloud, config)
    
    # Train network
    start_time = time.time()
    trainer.train()
    logger.info(f"Training took {time.time() - start_time:.2f} seconds")
    
    # Extract mesh
    mesh = trainer.extract_mesh()
    
    # TODO: Add texture mapping from images using cameras
    
    return mesh


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural implicit surface reconstruction")
    parser.add_argument("--pointcloud", required=True, help="Path to input pointcloud (.ply)")
    parser.add_argument("--output", required=True, help="Output mesh path (.ply)")
    parser.add_argument("--resolution", type=int, default=128, help="Marching cubes resolution")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    
    args = parser.parse_args()
    
    # Load pointcloud
    pointcloud = o3d.io.read_point_cloud(args.pointcloud)
    
    # Configure trainer
    config = {
        'marching_cubes_resolution': args.resolution,
        'num_iterations': args.iterations
    }
    
    # Run optimization
    mesh = optimize_neural_surface(pointcloud, None, None, config)
    
    # Save mesh
    o3d.io.write_triangle_mesh(args.output, mesh)
    
    print(f"Mesh saved to {args.output}")
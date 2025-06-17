# =============================================================================
# Spiral Dataset Class for Multi-Class Classification
# =============================================================================

import numpy as np  # Import numpy for numerical operations
import torch  # Import PyTorch for tensor operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from torch.utils.data import Dataset  # Import Dataset base class

class SpiralData(Dataset):  # Custom dataset class for spiral data
    """Generate spiral dataset for 3-class classification"""
    
    def __init__(self, K=3, N=500):  # Constructor with number of classes and samples per class
        """
        Create spiral dataset with K classes and N samples per class
        Modified from: http://cs231n.github.io/neural-networks-case-study/
        """
        D = 2  # Number of dimensions (2D spiral)
        X = np.zeros((N * K, D))  # Initialize data matrix
        y = np.zeros(N * K, dtype='uint8')  # Initialize class labels
        
        # Generate spiral data for each class
        for j in range(K):  # Loop through each class
            ix = range(N * j, N * (j + 1))  # Index range for current class
            r = np.linspace(0.0, 1, N)  # Radius values from 0 to 1
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # Theta with noise
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # Convert to Cartesian coordinates
            y[ix] = j  # Set class labels
        
        # Convert to PyTorch tensors
        self.y = torch.from_numpy(y).type(torch.LongTensor)  # Labels as long tensor
        self.x = torch.from_numpy(X).type(torch.FloatTensor)  # Features as float tensor
        self.len = y.shape[0]  # Store dataset length
        self.K = K  # Store number of classes
        self.N = N  # Store samples per class
    
    def __getitem__(self, index):  # Get single sample
        """Return feature-label pair for given index"""
        return self.x[index], self.y[index]  # Return data point
    
    def __len__(self):  # Get dataset length
        """Return total number of samples"""
        return self.len  # Return dataset size
    
    def plot_data(self):  # Visualize the dataset
        """Plot the spiral dataset with different colors for each class"""
        plt.figure(figsize=(10, 8))  # Set figure size
        
        # Plot each class with different colors and markers
        colors = ['red', 'green', 'blue']  # Colors for each class
        markers = ['o', 's', '^']  # Different markers for each class
        
        for class_idx in range(self.K):  # Loop through classes
            mask = self.y == class_idx  # Create mask for current class
            plt.scatter(self.x[mask, 0].numpy(), self.x[mask, 1].numpy(),                       
                       c=colors[class_idx], marker=markers[class_idx], s=50,
                       label=f'Class {class_idx}', alpha=0.7)
        
        plt.title("3-Class Spiral Dataset", fontsize=14)  # Add title
        plt.xlabel("Feature 1", fontsize=12)  # X-axis label
        plt.ylabel("Feature 2", fontsize=12)  # Y-axis label
        plt.legend(fontsize=12)  # Add legend
        plt.grid(True, alpha=0.3)  # Add grid
        plt.axis('equal')  # Equal aspect ratio
        plt.show()  # Display plot
"""
Clean MNIST test set: Convert from idx binary format to CSV.
- X: label (digit 0-9)
- Y: 784 pixel values (28x28 flattened)
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA


def load_idx_images(filepath: str) -> np.ndarray:
    """
    Load MNIST images from idx3-ubyte format.
    
    Returns
    -------
    images : np.ndarray, shape (n_samples, 784)
        Flattened image data (28x28 -> 784D).
    """
    with open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read image data
        image_data = np.fromfile(f, dtype=np.uint8, count=n_images * n_rows * n_cols)
        images = image_data.reshape(n_images, n_rows * n_cols)
    
    return images


def load_idx_labels(filepath: str) -> np.ndarray:
    """
    Load MNIST labels from idx1-ubyte format.
    
    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Label data (0-9).
    """
    with open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read label data
        labels = np.fromfile(f, dtype=np.uint8, count=n_labels)
    
    return labels


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw_mnist"
    clean_dir = project_root / "data" / "clean_mnist"
    
    # Create output directory if it doesn't exist
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test set
    print("Loading MNIST test set...")
    test_images = load_idx_images(str(raw_dir / "t10k-images-idx3-ubyte"))
    test_labels = load_idx_labels(str(raw_dir / "t10k-labels-idx1-ubyte"))
    
    print(f"  Images shape: {test_images.shape}")
    print(f"  Labels shape: {test_labels.shape}")
    
    # Normalize pixel values to [0, 1]
    print("Normalizing pixel values...")
    test_images_normalized = test_images.astype(np.float32) / 255.0
    
    # Z-score normalize across all pixels (center and scale)
    print("Z-score normalizing...")
    pixel_mean = test_images_normalized.mean()
    pixel_std = test_images_normalized.std()
    test_images_normalized = (test_images_normalized - pixel_mean) / pixel_std
    
    # Apply PCA to reduce dimensionality
    print("Applying PCA to reduce dimensionality to 100...")
    pca = PCA(n_components=100)
    test_images_pca = pca.fit_transform(test_images_normalized)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Reduced shape: {test_images_pca.shape}")
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame(test_images_pca)
    df.insert(0, 'X', test_labels)
    
    # Rename columns to Y_0, Y_1, ..., Y_99 (100 PCA components)
    pixel_cols = [f'Y_{i}' for i in range(100)]
    df.columns = ['X'] + pixel_cols
    
    print(f"  DataFrame shape: {df.shape}")
    
    # Save to CSV
    output_path = clean_dir / "mnist_test.csv"
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"✓ Done! CSV saved with shape {df.shape}")
    print(f"  Columns: X (label) + Y_0 to Y_99 (PCA components)")
    
    # Verify
    df_verify = pd.read_csv(output_path, nrows=5)
    print(f"\nFirst 5 rows (showing first 10 columns):")
    print(df_verify.iloc[:, :10])


if __name__ == "__main__":
    main()

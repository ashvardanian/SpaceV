#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["numpy", "tqdm"]
# ///
"""
Sample 100M vectors from the 1B SpaceV dataset, ensuring all ground truth IDs are included.
"""
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_matrix_header(file_path):
    """Load the header (rows, cols) from a binary matrix file."""
    with open(file_path, "rb") as f:
        rows = struct.unpack("<I", f.read(4))[0]
        cols = struct.unpack("<I", f.read(4))[0]
    return rows, cols


def load_matrix(file_path, dtype=np.int8):
    """Load a binary matrix file."""
    with open(file_path, "rb") as f:
        rows = struct.unpack("<I", f.read(4))[0]
        cols = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=dtype).reshape(rows, cols)
    return data


def save_matrix(file_path, matrix):
    """Save a matrix to binary format with header."""
    with open(file_path, "wb") as f:
        rows, cols = matrix.shape
        f.write(struct.pack("<I", rows))
        f.write(struct.pack("<I", cols))
        f.write(matrix.tobytes())


def main():
    print("Loading ground truth to find required vector IDs...")

    # Load ground truth to get all referenced vector IDs
    groundtruth = load_matrix("groundtruth.30K.i32bin", dtype=np.int32)
    print(f"Loaded ground truth: {groundtruth.shape}")

    # Get all unique IDs referenced in ground truth
    required_ids = np.unique(groundtruth.flatten())
    print(f"Found {len(required_ids):,} unique vector IDs in ground truth")
    print(f"ID range: {required_ids.min():,} to {required_ids.max():,}")

    # Load full dataset header to understand size
    total_vectors, vector_dim = load_matrix_header("base.1B.i8bin")
    print(f"Full dataset: {total_vectors:,} vectors, {vector_dim} dimensions")

    # Calculate how many additional random vectors we need
    target_size = 100_000_000  # 100M
    additional_needed = target_size - len(required_ids)
    print(
        f"Need {additional_needed:,} additional random vectors to reach {target_size:,}"
    )

    # Create set of all possible IDs excluding the required ones
    all_ids = set(range(total_vectors))
    available_ids = list(all_ids - set(required_ids))

    # Sample additional random IDs
    np.random.seed(42)  # For reproducibility
    additional_ids = np.random.choice(
        available_ids, size=additional_needed, replace=False
    )

    # Combine required and additional IDs, then sort
    final_ids = np.concatenate([required_ids, additional_ids])
    final_ids = np.sort(final_ids)

    print(f"Final sample: {len(final_ids):,} vectors")
    print(f"ID range: {final_ids.min():,} to {final_ids.max():,}")

    # Load and sample the vectors
    print("Loading full dataset and sampling vectors...")
    sampled_vectors = []
    
    with open("base.1B.i8bin", "rb") as f:
        # Skip header
        f.read(8)
        
        # Load vectors with progress bar
        with tqdm(final_ids, desc="Sampling vectors") as pbar:
            for vector_id in pbar:
                # Seek to the vector position
                f.seek(8 + vector_id * vector_dim)
                vector_data = f.read(vector_dim)
                vector = np.frombuffer(vector_data, dtype=np.int8)
                sampled_vectors.append(vector)

    # Convert to numpy array
    sampled_matrix = np.array(sampled_vectors, dtype=np.int8)
    print(f"Sampled matrix shape: {sampled_matrix.shape}")

    print("Saving sampled dataset...")
    save_matrix("base.100M.i8bin", sampled_matrix)

    # Save the vector IDs as well
    print("Saving vector IDs...")
    # Reshape final_ids to be a column vector (100M x 1)
    ids_matrix = final_ids.reshape(-1, 1).astype(np.int32)
    save_matrix("ids.100M.i32bin", ids_matrix)

    # Verify the files were created correctly
    verification_rows, verification_cols = load_matrix_header("base.100M.i8bin")
    print(
        f"Verification: saved {verification_rows:,} vectors with {verification_cols} dimensions"
    )

    ids_rows, ids_cols = load_matrix_header("ids.100M.i32bin")
    print(f"Verification: saved {ids_rows:,} vector IDs with {ids_cols} column(s)")

    base_file_size = Path("base.100M.i8bin").stat().st_size / (1024**3)
    ids_file_size = Path("ids.100M.i32bin").stat().st_size / (1024**2)
    print(f"Base file size: {base_file_size:.2f} GB")
    print(f"IDs file size: {ids_file_size:.2f} MB")

    print("Dataset sampling complete!")


if __name__ == "__main__":
    main()

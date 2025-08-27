#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["numpy", "tqdm", "usearch>=2.20.1"]
# ///
"""
Sample 100M vectors from the 1B SpaceV dataset, ensuring all ground truth IDs are included.
To run via `uv`:

    uv run --script sample.py # Sample and verify
    uv run --script sample.py --verify-only # Only verify existing sample
"""
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from usearch.io import load_matrix, save_matrix


def print_header(title):
    """Print a clean section header."""
    width = 60
    print()
    print("═" * width)
    print(f"{title:^{width}}")
    print("═" * width)


def sample_dataset():
    """Sample 100M vectors from the 1B dataset, ensuring all ground truth vectors are included."""
    
    print_header("SAMPLING PHASE")
    
    # Step 1: Load ground truth
    print("\n1. Loading ground truth to find required vector IDs...")
    if not os.path.exists("groundtruth.30K.i32bin"):
        raise FileNotFoundError("   ✗ groundtruth.30K.i32bin not found. Please download it from s3://bigger-ann/spacev-1b/")
    
    groundtruth = load_matrix("groundtruth.30K.i32bin")
    print(f"   ✓ Loaded ground truth: {groundtruth.shape}")
    
    required_ids = np.unique(groundtruth.flatten())
    print(f"   ✓ Found {len(required_ids):,} unique vector IDs in ground truth")
    print(f"   ✓ ID range: {required_ids.min():,} to {required_ids.max():,}")
    
    # Step 2: Read dataset dimensions
    print("\n2. Reading dataset dimensions...")
    if not os.path.exists("base.1B.i8bin"):
        raise FileNotFoundError("   ✗ base.1B.i8bin not found. Please ensure it's in the current directory")
    
    base_matrix_view = load_matrix("base.1B.i8bin", view=True)
    total_vectors, vector_dim = base_matrix_view.shape
    print(f"   ✓ Full dataset: {total_vectors:,} vectors, {vector_dim} dimensions")
    del base_matrix_view
    
    # Step 3: Calculate sampling strategy
    print("\n3. Calculating sampling strategy...")
    target_size = 100_000_000  # 100M
    additional_needed = target_size - len(required_ids)
    print(f"   ✓ Target size: {target_size:,} vectors")
    print(f"   ✓ Need {additional_needed:,} additional random vectors")
    
    # Step 4: Sample additional IDs
    print("\n4. Sampling additional vector IDs...")
    np.random.seed(42)  # For reproducibility
    
    candidates_needed = int(additional_needed * 1.2)  # 20% extra for collision handling
    print(f"   → Generating {candidates_needed:,} candidate IDs...")
    candidates = np.random.choice(total_vectors, size=candidates_needed, replace=False)
    
    print(f"   → Removing collisions with ground truth IDs...")
    additional_ids = np.setdiff1d(candidates, required_ids)[:additional_needed]
    
    # Combine and sort final IDs
    final_ids = np.concatenate([required_ids, additional_ids])
    final_ids = np.sort(final_ids)
    
    print(f"   ✓ Final sample: {len(final_ids):,} vectors")
    print(f"   ✓ ID range: {final_ids.min():,} to {final_ids.max():,}")
    
    # Step 5: Sample vectors
    print("\n5. Loading and sampling vectors...")
    base_matrix = load_matrix("base.1B.i8bin", view=True)
    
    sampled_matrix = np.empty((len(final_ids), vector_dim), dtype=np.int8)
    with tqdm(total=len(final_ids), desc="   Sampling vectors") as pbar:
        batch_size = 10000
        for i in range(0, len(final_ids), batch_size):
            end = min(i + batch_size, len(final_ids))
            sampled_matrix[i:end] = base_matrix[final_ids[i:end]]
            pbar.update(end - i)
    
    print(f"   ✓ Sampled matrix shape: {sampled_matrix.shape}")
    
    # Step 6: Save sampled dataset
    print("\n6. Saving sampled dataset...")
    save_matrix(sampled_matrix, "base.100M.i8bin")
    print("   ✓ Saved base.100M.i8bin")
    
    # Save vector IDs
    ids_matrix = final_ids.reshape(-1, 1).astype(np.int32)
    save_matrix(ids_matrix, "ids.100M.i32bin")
    print("   ✓ Saved ids.100M.i32bin")
    
    return final_ids, required_ids


def verify_dataset(final_ids=None, required_ids=None):
    """Verify the integrity of the sampled dataset."""
    
    print_header("VERIFICATION PHASE")
    
    # Step 1: Load saved files
    print("\n1. Loading saved files as memory-mapped views...")
    
    if not os.path.exists("base.100M.i8bin") or not os.path.exists("ids.100M.i32bin"):
        print("   ✗ Sampled files not found. Please run sampling first.")
        return False
    
    saved_base = load_matrix("base.100M.i8bin", view=True)
    saved_ids = load_matrix("ids.100M.i32bin", view=True)
    original_base = load_matrix("base.1B.i8bin", view=True)
    
    print(f"   ✓ Saved vectors shape: {saved_base.shape}")
    print(f"   ✓ Saved IDs shape: {saved_ids.shape}")
    print(f"   ✓ Original dataset shape: {original_base.shape}")
    
    # Extract saved ID values
    saved_id_values = saved_ids.flatten()
    
    # If we don't have final_ids/required_ids (verify-only mode), reconstruct them
    if final_ids is None:
        print("\n2. Reconstructing sampling information...")
        final_ids = saved_id_values
        
        # Load ground truth to get required_ids
        if os.path.exists("groundtruth.30K.i32bin"):
            groundtruth = load_matrix("groundtruth.30K.i32bin")
            required_ids = np.unique(groundtruth.flatten())
            print(f"   ✓ Loaded {len(required_ids):,} ground truth IDs")
        else:
            print("   ⚠ Ground truth file not found, skipping ground truth verification")
            required_ids = None
    
    # Step 2: Dimension checks
    print("\n3. Dimension checks...")
    vector_dim = original_base.shape[1]
    
    try:
        assert saved_base.shape[0] == len(final_ids), f"Vector count mismatch: {saved_base.shape[0]} != {len(final_ids)}"
        assert saved_base.shape[1] == vector_dim, f"Vector dimension mismatch: {saved_base.shape[1]} != {vector_dim}"
        assert saved_ids.shape[0] == len(final_ids), f"ID count mismatch: {saved_ids.shape[0]} != {len(final_ids)}"
        print("   ✓ All dimensions match expected values")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False
    
    # Step 3: Verify IDs match
    print("\n4. Verifying saved IDs consistency...")
    if np.array_equal(saved_id_values, final_ids):
        print("   ✓ All IDs correctly saved and ordered")
    else:
        print("   ✗ Saved IDs don't match expected sampling IDs")
        return False
    
    # Step 4: Random sampling verification
    print("\n5. Random sampling verification...")
    np.random.seed(123)
    num_checks = min(100, len(final_ids))
    random_indices = np.random.choice(len(final_ids), size=num_checks, replace=False)
    
    mismatches = 0
    with tqdm(random_indices, desc="   Verifying vectors") as pbar:
        for i in pbar:
            original_idx = final_ids[i]
            saved_vector = saved_base[i]
            original_vector = original_base[original_idx]
            
            if not np.array_equal(saved_vector, original_vector):
                mismatches += 1
                if mismatches <= 3:
                    print(f"\n   ⚠ Mismatch at index {i} (original index {original_idx})")
    
    if mismatches == 0:
        print(f"   ✓ All {num_checks} randomly checked vectors match perfectly")
    else:
        print(f"   ✗ Found {mismatches}/{num_checks} mismatches")
        return False
    
    # Step 5: File size verification
    print("\n6. File size verification...")
    base_file_size = Path("base.100M.i8bin").stat().st_size / (1024**3)
    ids_file_size = Path("ids.100M.i32bin").stat().st_size / (1024**2)
    expected_base_size = (8 + saved_base.shape[0] * saved_base.shape[1] * 1) / (1024**3)
    expected_ids_size = (8 + saved_ids.shape[0] * saved_ids.shape[1] * 4) / (1024**2)
    
    print(f"   → Base file: {base_file_size:.3f} GB (expected: {expected_base_size:.3f} GB)")
    print(f"   → IDs file: {ids_file_size:.3f} MB (expected: {expected_ids_size:.3f} MB)")
    
    if abs(base_file_size - expected_base_size) < 0.001 and abs(ids_file_size - expected_ids_size) < 0.001:
        print("   ✓ File sizes match expected values")
    else:
        print("   ⚠ File sizes differ slightly from expected")
    
    return True


def main():
    """Main entry point for the sampling script."""
    
    # Check command line arguments
    verify_only = "--verify-only" in sys.argv
    
    # Check if sampled files already exist
    sample_exists = os.path.exists("base.100M.i8bin") and os.path.exists("ids.100M.i32bin")
    
    if verify_only:
        if not sample_exists:
            print("Error: Cannot verify - sampled files don't exist yet.")
            print("Please run without --verify-only flag to create the sample first.")
            sys.exit(1)
        
        print("Running verification only...")
        success = verify_dataset()
    else:
        if sample_exists:
            print("Sampled files already exist!")
            response = input("Do you want to: [s]kip to verification, [r]e-sample, or [q]uit? ").lower().strip()
            if response == 'q':
                print("Exiting.")
                sys.exit(0)
            elif response == 's':
                verify_only = True
        
        if verify_only:
            success = verify_dataset()
        else:
            # Run sampling then verification
            final_ids, required_ids = sample_dataset()
            success = verify_dataset(final_ids, required_ids)
    
    # Final status
    if success:
        print_header("✓ DATASET SAMPLING AND VERIFICATION COMPLETE!")
    else:
        print_header("✗ VERIFICATION FAILED")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

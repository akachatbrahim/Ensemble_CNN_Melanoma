"""
Memory-optimized test script for handling OOM errors
"""
import tensorflow as tf
import numpy as np
import gc

def test_memory_optimized():
    # === STEP 1: Enable mixed precision training (uses float16 to save memory) ===
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision enabled (float16 for forward pass, float32 for variables)")
    except Exception as e:
        print(f"Mixed precision not available: {e}")

    # === STEP 2: Configure GPU memory growth ===
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    # === STEP 3: Clear memory before loading data ===
    gc.collect()
    print("✓ Memory cleaned")
    

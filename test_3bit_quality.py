#!/usr/bin/env python3
"""
Test 3-bit conversion on a small sample.
"""
import struct
import numpy as np

def test_conversion():
    """Test 3-bit quantization quality."""
    print("Testing 3-bit quantization quality\n")
    
    # Simulate some weight values (typical range for LLM weights)
    np.random.seed(42)
    test_values = np.random.randn(10000) * 0.5  # Mean 0, std 0.5
    
    print(f"Original: min={test_values.min():.4f}, max={test_values.max():.4f}, std={test_values.std():.4f}")
    
    # 2-bit ternary
    ternary_lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    q2 = np.where(test_values > 0.3, 1, np.where(test_values < -0.3, -1, 0))
    q2_packed = np.where(q2 == -1, 0, np.where(q2 == 1, 2, 1)).astype(np.uint8)
    print(f"\n2-bit ternary: unique values={np.unique(q2)}, MSE={np.mean((test_values - q2)**2):.6f}")
    
    # 3-bit
    clipped = np.clip(test_values * 3.5, -3.5, 3.5)
    q3 = np.round(clipped + 3.5).astype(np.uint8)
    q3 = np.clip(q3, 0, 7)
    q3_dequant = (q3.astype(np.float32) - 3.5) / 3.5
    print(f"3-bit: unique values={np.unique(q3)}, MSE={np.mean((test_values - q3_dequant)**2):.6f}")
    
    # 4-bit
    clipped4 = np.clip(test_values, -7.5, 7.5)
    q4 = np.round(clipped4 + 7.5).astype(np.uint8)
    q4 = np.clip(q4, 0, 15)
    q4_dequant = q4.astype(np.float32) - 7.5
    print(f"4-bit: unique values={len(np.unique(q4))}, MSE={np.mean((test_values - q4_dequant)**2):.6f}")
    
    print("\n" + "="*50)
    print("Conclusion: 3-bit should give ~4x better MSE than 2-bit")
    print("="*50)

if __name__ == '__main__':
    test_conversion()

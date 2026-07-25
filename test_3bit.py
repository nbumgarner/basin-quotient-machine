#!/usr/bin/env python3
"""
Test 3-bit conversion on first few tensors only.
"""
import struct
import numpy as np
import sys

def test_3bit_conversion(input_path, output_path, max_tensors=5):
    """Test 3-bit conversion on first few tensors."""
    print(f"Testing 3-bit conversion: {input_path}")
    
    # Read input header
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        d_model = struct.unpack('<I', f.read(4))[0]
        ffn_dim = struct.unpack('<I', f.read(4))[0]
        n_layers = struct.unpack('<I', f.read(4))[0]
        n_heads = struct.unpack('<I', f.read(4))[0]
        n_kv_heads = struct.unpack('<I', f.read(4))[0]
        vocab_size = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<I', f.read(4))[0]
        
        print(f"Header: d={d_model}, layers={n_layers}, vocab={vocab_size}")
        print(f"Total tensors: {n_tensors}, testing first {max_tensors}")
    
    # Open output file
    with open(output_path, 'wb') as out:
        # Write 3-bit header
        out.write(struct.pack('<4sIIIIIII',
            b'BQSM',
            3,        # version (3-bit)
            d_model,
            ffn_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size
        ))
        out.write(struct.pack('<I', max_tensors))  # Only test with max_tensors
        
        # Process tensors
        with open(input_path, 'rb') as inp:
            inp.read(36)  # Skip header
            
            for i in range(max_tensors):
                name_len = struct.unpack('<H', inp.read(2))[0]
                name = inp.read(name_len).decode('utf-8')
                
                ndims = struct.unpack('<B', inp.read(1))[0]
                shape = []
                for _ in range(ndims):
                    dim = struct.unpack('<Q', inp.read(8))[0]
                    shape.append(dim)
                
                orig_n = struct.unpack('<I', inp.read(4))[0]
                ttype = struct.unpack('<B', inp.read(1))[0]
                data_n = struct.unpack('<I', inp.read(4))[0]
                
                data = inp.read(data_n)
                
                print(f"\nTensor {i}: {name}")
                print(f"  Shape: {shape}, type: {ttype}, data_n: {data_n}")
                
                # Convert
                if len(data) > 0 and data_n > 0 and ttype != 1:
                    # Show original stats
                    float_values = dequantize_2bit(data)
                    print(f"  Original: min={float_values.min():.3f}, max={float_values.max():.3f}, mean={float_values.mean():.3f}")
                    
                    packed_3bit = quantize_to_3bit(float_values)
                    
                    # Show converted stats
                    converted = dequantize_3bit(packed_3bit)
                    print(f"  Converted: min={converted.min():.3f}, max={converted.max():.3f}, mean={converted.mean():.3f}")
                    print(f"  Error: MSE={np.mean((float_values - converted)**2):.6f}")
                else:
                    packed_3bit = data
                    print(f"  Skipping (F32 or empty)")
                
                # Write
                out.write(struct.pack('<H', name_len))
                out.write(name.encode('utf-8'))
                out.write(struct.pack('<B', ndims))
                for dim in shape:
                    out.write(struct.pack('<Q', dim))
                out.write(struct.pack('<I', orig_n))
                out.write(struct.pack('<B', ttype))
                out.write(struct.pack('<I', len(packed_3bit)))
                out.write(packed_3bit)
    
    print(f"\nTest complete! Output: {output_path}")
    return True

def dequantize_2bit(data):
    """Dequantize 2-bit ternary."""
    ternary_lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    values = np.empty(len(data_arr) * 4, dtype=np.float32)
    for shift in [0, 2, 4, 6]:
        idx = np.arange(len(data_arr)) * 4 + (shift // 2)
        nb = (data_arr >> shift) & 3
        values[idx] = ternary_lut[nb]
    return values

def quantize_to_3bit(float_values):
    """Quantize float to 3-bit."""
    clipped = np.clip(float_values * 3.5, -3.5, 3.5)
    q3 = np.round(clipped + 3.5).astype(np.uint8)
    q3 = np.clip(q3, 0, 7)
    packed = np.zeros(len(q3) // 2, dtype=np.uint8)
    for i in range(0, len(q3), 2):
        packed[i//2] = (q3[i] << 4) | q3[i+1]
    return packed

def dequantize_3bit(packed):
    """Dequantize 3-bit."""
    q3 = np.empty(len(packed) * 2, dtype=np.uint8)
    for i in range(len(packed)):
        q3[i*2] = (packed[i] >> 4) & 0x0F
        q3[i*2+1] = packed[i] & 0x0F
    # Mask to 3 bits
    q3 = q3 & 7
    lut = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    return lut[q3]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: test_3bit.py <input.bqsm> <output.bqsm>")
        sys.exit(1)
    
    test_3bit_conversion(sys.argv[1], sys.argv[2], max_tensors=5)

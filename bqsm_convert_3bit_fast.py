#!/usr/bin/env python3
"""
Fast 2-bit to 3-bit BQSM converter using vectorized numpy operations.
"""
import struct
import numpy as np
import sys
import os

def convert_fast(input_path, output_path):
    """Convert 2-bit BQSM to 3-bit BQSM using vectorized operations."""
    print(f"Converting: {input_path} -> {output_path}")
    
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
        
        print(f"Header: d={d_model}, layers={n_layers}, vocab={vocab_size}, tensors={n_tensors}")
    
    # Open output file
    with open(output_path, 'wb') as out:
        # Write 3-bit header
        out.write(struct.pack('<4sIIIIIII', b'BQSM', 3, d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size))
        out.write(struct.pack('<I', n_tensors))
        
        # Process tensors
        with open(input_path, 'rb') as inp:
            inp.read(36)
            
            for i in range(n_tensors):
                name_len = struct.unpack('<H', inp.read(2))[0]
                name = inp.read(name_len).decode('utf-8')
                
                ndims = struct.unpack('<B', inp.read(1))[0]
                shape = [struct.unpack('<Q', inp.read(8))[0] for _ in range(ndims)]
                
                orig_n = struct.unpack('<I', inp.read(4))[0]
                ttype = struct.unpack('<B', inp.read(1))[0]
                data_n = struct.unpack('<I', inp.read(4))[0]
                
                data = inp.read(data_n)
                
                if len(data) > 0 and data_n > 0 and ttype != 1:
                    packed_3bit = convert_2bit_to_3bit_vectorized(data)
                else:
                    packed_3bit = data
                
                out.write(struct.pack('<H', name_len))
                out.write(name.encode('utf-8'))
                out.write(struct.pack('<B', ndims))
                for dim in shape:
                    out.write(struct.pack('<Q', dim))
                out.write(struct.pack('<I', orig_n))
                out.write(struct.pack('<B', ttype))
                out.write(struct.pack('<I', len(packed_3bit)))
                out.write(packed_3bit)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{n_tensors} tensors...")
    
    print(f"\nDone! Size: {os.path.getsize(output_path):,} bytes")
    return True

def convert_2bit_to_3bit_vectorized(data):
    """Fully vectorized 2-bit to 3-bit conversion."""
    # Unpack 2-bit to float32
    data_arr = np.frombuffer(data, dtype=np.uint8)
    n_values = len(data_arr) * 4
    
    # Create indices for unpacking
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    idx_base = np.arange(len(data_arr)).reshape(-1, 1) * 4
    idx = idx_base + np.array([0, 1, 2, 3]).reshape(1, 4)
    
    # Unpack all 4 values per byte
    nb = (data_arr.reshape(-1, 1) >> shifts) & 3
    ternary_lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    values = ternary_lut[nb].ravel()
    
    # Quantize to 3-bit
    clipped = np.clip(values * 3.5, -3.5, 3.5)
    q3 = np.round(clipped + 3.5).astype(np.uint8)
    q3 = np.clip(q3, 0, 7)
    
    # Pack 2 values per byte (vectorized)
    n_packed = len(q3) // 2
    q3_reshaped = q3[:n_packed*2].reshape(n_packed, 2)
    packed = (q3_reshaped[:, 0:1] << 4 | q3_reshaped[:, 1:2]).astype(np.uint8).ravel()
    
    return packed.tobytes()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_convert_3bit_fast.py <input.bqsm> <output.bqsm>")
        sys.exit(1)
    
    convert_fast(sys.argv[1], sys.argv[2])

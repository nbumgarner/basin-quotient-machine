#!/usr/bin/env python3
"""
Direct 2-bit to 3-bit converter - optimized for speed.
Uses numpy vectorized ops, no intermediate float arrays.
"""
import struct
import numpy as np
import sys
import os

def convert(input_path, output_path):
    """Convert 2-bit BQSM to 3-bit BQSM."""
    print(f"Converting: {input_path} -> {output_path}")
    
    # Read header
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
        
        print(f"d={d_model}, layers={n_layers}, vocab={vocab_size}, tensors={n_tensors}")
    
    # Process
    with open(output_path, 'wb') as out:
        out.write(struct.pack('<4sIIIIIII', b'BQSM', 5, d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size))
        out.write(struct.pack('<I', n_tensors))
        
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
                    # Fast conversion
                    packed = convert_2bit_to_3bit(data)
                else:
                    packed = data
                
                out.write(struct.pack('<H', name_len))
                out.write(name.encode('utf-8'))
                out.write(struct.pack('<B', ndims))
                for dim in shape:
                    out.write(struct.pack('<Q', dim))
                out.write(struct.pack('<I', orig_n))
                out.write(struct.pack('<B', ttype))
                out.write(struct.pack('<I', len(packed)))
                out.write(packed)
                
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{n_tensors} tensors...")
    
    print(f"Done! {os.path.getsize(output_path):,} bytes")

def convert_2bit_to_3bit(data):
    """Fast 2-bit to 3-bit conversion using lookup tables."""
    # Precompute: 2-bit values (-1, 0, 1) -> 3-bit packed
    # 2-bit: 0=-1, 1=0, 2=1, 3=0
    # Target: 0=-3.5, 1=-2.5, 2=-1.5, 3=-0.5, 4=0.5, 5=1.5, 6=2.5, 7=3.5
    
    arr = np.frombuffer(data, dtype=np.uint8)
    
    # Unpack: each byte -> 4 x 2-bit values (0-3)
    # Then map to 3-bit values (0-7) and pack 2 per byte
    
    # Step 1: Unpack 2-bit values (vectorized)
    # Create 4 arrays for each bit position
    v0 = (arr >> 0) & 3  # bits 0,1
    v1 = (arr >> 2) & 3  # bits 2,3
    v2 = (arr >> 4) & 3  # bits 4,5
    v3 = (arr >> 6) & 3  # bits 6,7
    
    # Step 2: Map 2-bit to float, then quantize to 3-bit
    # 2-bit: 0=-1, 1=0, 2=1, 3=0
    # Scale by 3.5, round, add 3.5 to get 0-7
    
    lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    
    # Convert all 4 positions at once
    f0 = lut[v0].astype(np.int8) * 3.5 + 3.5
    f1 = lut[v1].astype(np.int8) * 3.5 + 3.5
    f2 = lut[v2].astype(np.int8) * 3.5 + 3.5
    f3 = lut[v3].astype(np.int8) * 3.5 + 3.5
    
    # Round and clip
    q0 = np.clip(np.round(f0).astype(np.uint8), 0, 7)
    q1 = np.clip(np.round(f1).astype(np.uint8), 0, 7)
    q2 = np.clip(np.round(f2).astype(np.uint8), 0, 7)
    q3 = np.clip(np.round(f3).astype(np.uint8), 0, 7)
    
    # Pack 2 values per byte (high 4 bits, low 4 bits)
    # Result: byte = (q_high << 4) | q_low
    packed = ((q0 << 4) | q1).astype(np.uint8)
    packed2 = ((q2 << 4) | q3).astype(np.uint8)
    
    # Interleave
    result = np.empty(len(arr) * 2, dtype=np.uint8)
    result[0::2] = packed
    result[1::2] = packed2
    
    return result.tobytes()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_2to3.py <input.bqsm> <output.bqsm>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""
Fast 2-bit to 4-bit BQSM converter using vectorized operations.
Processes tensors in chunks to avoid OOM and speed up conversion.
"""
import struct
import numpy as np
import sys
import os

def convert_fast(input_path, output_path, chunk_size=10000):
    """Convert 2-bit BQSM to 4-bit BQSM with chunked processing."""
    print(f"Converting: {input_path} -> {output_path}")
    
    # Read input header
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'BQSM':
            raise ValueError("Not a BQSM file")
        
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
        # Write 4-bit header
        out.write(struct.pack('<4sIIIIIII',
            b'BQSM',  # magic
            4,        # version (4-bit)
            d_model,
            ffn_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size
        ))
        out.write(struct.pack('<I', n_tensors))
        
        # Process tensors one at a time
        with open(input_path, 'rb') as inp:
            # Skip header (36 bytes)
            inp.read(36)
            
            for i in range(n_tensors):
                # Read tensor metadata
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
                
                # Read data
                data = inp.read(data_n)
                
                # Convert 2-bit to 4-bit
                if len(data) > 0 and data_n > 0 and ttype != 1:  # ttype=1 is F32, skip
                    # Vectorized conversion
                    packed_4bit = convert_2bit_to_4bit_vectorized(data)
                else:
                    packed_4bit = data
                
                # Write tensor to output
                out.write(struct.pack('<H', name_len))
                out.write(name.encode('utf-8'))
                out.write(struct.pack('<B', ndims))
                for dim in shape:
                    out.write(struct.pack('<Q', dim))
                out.write(struct.pack('<I', orig_n))
                out.write(struct.pack('<B', ttype))
                out.write(struct.pack('<I', len(packed_4bit)))
                out.write(packed_4bit)
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{n_tensors} tensors...")
                
                # Free memory
                del data
                del shape
        
        print(f"\nConversion complete!")
        size = os.path.getsize(output_path)
        print(f"  Output: {output_path}")
        print(f"  Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    
    return True

def convert_2bit_to_4bit_vectorized(data):
    """Vectorized 2-bit to 4-bit conversion."""
    # Ternary LUT: 0=-1, 1=0, 2=1, 3=0
    ternary_lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    
    # Unpack 4 values per byte (vectorized)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    values = np.empty(len(data_arr) * 4, dtype=np.float32)
    
    for shift in [0, 2, 4, 6]:
        idx = np.arange(len(data_arr)) * 4 + (shift // 2)
        nb = (data_arr >> shift) & 3
        values[idx] = ternary_lut[nb]
    
    # Quantize to 4-bit
    clipped = np.clip(values, -7.5, 7.5)
    q4 = np.round(clipped + 7.5).astype(np.uint8)
    q4 = np.clip(q4, 0, 15)
    
    # Pack 2 values per byte
    packed = np.zeros(len(q4) // 2, dtype=np.uint8)
    for i in range(0, len(q4), 2):
        packed[i//2] = (q4[i] << 4) | q4[i+1]
    
    return packed.tobytes()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_convert_4bit_fast.py <input_2bit.bqsm> <output_4bit.bqsm>")
        sys.exit(1)
    
    success = convert_fast(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)

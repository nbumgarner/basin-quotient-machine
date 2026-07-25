#!/usr/bin/env python3
"""
Efficient 2-bit to 3-bit BQSM converter.
Processes tensors in chunks to avoid OOM.
"""
import struct
import numpy as np
import sys
import os

def convert_2bit_to_3bit(input_path, output_path, chunk_elements=1000000):
    """Convert 2-bit BQSM to 3-bit BQSM with chunked processing."""
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
        out.write(struct.pack('<I', n_tensors))
        
        # Process tensors one at a time
        with open(input_path, 'rb') as inp:
            inp.read(36)  # Skip header
            
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
                
                # Convert 2-bit to 3-bit
                if len(data) > 0 and data_n > 0 and ttype != 1:
                    packed_3bit = convert_chunked(data, chunk_elements)
                else:
                    packed_3bit = data
                
                # Write tensor to output
                out.write(struct.pack('<H', name_len))
                out.write(name.encode('utf-8'))
                out.write(struct.pack('<B', ndims))
                for dim in shape:
                    out.write(struct.pack('<Q', dim))
                out.write(struct.pack('<I', orig_n))
                out.write(struct.pack('<B', ttype))
                out.write(struct.pack('<I', len(packed_3bit)))
                out.write(packed_3bit)
                
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

def convert_chunked(data, chunk_elements):
    """Convert 2-bit to 3-bit in chunks."""
    # Unpack 2-bit
    data_arr = np.frombuffer(data, dtype=np.uint8)
    n_values = len(data_arr) * 4
    
    # Process in chunks
    chunks = []
    chunk_size = chunk_elements // 4  # bytes
    
    for start in range(0, len(data_arr), chunk_size):
        end = min(start + chunk_size, len(data_arr))
        chunk = data_arr[start:end]
        
        # Unpack this chunk
        values = np.empty(len(chunk) * 4, dtype=np.float32)
        ternary_lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        
        for shift in [0, 2, 4, 6]:
            idx = np.arange(len(chunk)) * 4 + (shift // 2)
            nb = (chunk >> shift) & 3
            values[idx] = ternary_lut[nb]
        
        # Quantize to 3-bit
        clipped = np.clip(values * 3.5, -3.5, 3.5)
        q3 = np.round(clipped + 3.5).astype(np.uint8)
        q3 = np.clip(q3, 0, 7)
        
        # Pack 2 values per byte
        packed = np.zeros(len(q3) // 2, dtype=np.uint8)
        for j in range(0, len(q3), 2):
            packed[j//2] = (q3[j] << 4) | q3[j+1]
        
        chunks.append(packed)
    
    return np.concatenate(chunks).tobytes()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_convert_3bit_efficient.py <input_2bit.bqsm> <output_3bit.bqsm>")
        sys.exit(1)
    
    success = convert_2bit_to_3bit(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)

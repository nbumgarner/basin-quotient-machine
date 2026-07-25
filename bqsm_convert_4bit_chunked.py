#!/usr/bin/env python3
"""
Chunked 2-bit to 4-bit BQSM converter.
Processes tensors one at a time to avoid OOM.
"""
import struct
import numpy as np
import sys
import os

def convert_chunked(input_path, output_path, chunk_size=1):
    """Convert 2-bit BQSM to 4-bit BQSM, processing one tensor at a time."""
    print(f"Converting: {input_path} -> {output_path}")
    print(f"Processing {chunk_size} tensor(s) at a time\n")
    
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
        n_tensors = struct.unpack('<I', f.read(4))[0]  # 4 bytes, not 8
        
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
        out.write(struct.pack('<I', n_tensors))  # 4 bytes, not 8
        
        # Process tensors one at a time
        with open(input_path, 'rb') as inp:
            # Skip header (44 bytes)
            inp.read(44)
            
            for i in range(n_tensors):
                # Read tensor metadata
                name_len = struct.unpack('<Q', inp.read(8))[0]
                name = inp.read(name_len).decode('utf-8')
                
                tensor_type = struct.unpack('<I', inp.read(4))[0]
                ndims = struct.unpack('<I', inp.read(4))[0]
                shape = []
                for _ in range(ndims):
                    dim = struct.unpack('<Q', inp.read(8))[0]
                    shape.append(dim)
                
                data_len = struct.unpack('<Q', inp.read(8))[0]
                data = inp.read(data_len)
                
                # Dequantize 2-bit to float
                if len(data) > 0:
                    float_values = dequantize_2bit_to_float(data)
                    
                    # Quantize to 4-bit
                    packed_4bit = quantize_float_to_4bit(float_values)
                    
                    # Free memory
                    del float_values
                else:
                    packed_4bit = data
                
                # Write tensor to output
                name_bytes = name.encode('utf-8')
                out.write(struct.pack('<Q', len(name_bytes)))
                out.write(name_bytes)
                out.write(struct.pack('<I', tensor_type))
                out.write(struct.pack('<I', ndims))
                for dim in shape:
                    out.write(struct.pack('<Q', dim))
                out.write(struct.pack('<Q', len(packed_4bit)))
                out.write(packed_4bit)
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{n_tensors} tensors...")
                
                # Free memory
                del data
                del shape
        
        print(f"\nConversion complete!")
        print(f"  Output: {output_path}")
        
        # Get output file size
        size = os.path.getsize(output_path)
        print(f"  Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    
    return True

def dequantize_2bit_to_float(data):
    """Dequantize 2-bit ternary packed data to float32."""
    if len(data) == 0:
        return np.array([], dtype=np.float32)
    
    # Ternary LUT: 0=-1, 1=0, 2=1, 3=0
    ternary_lut = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    
    # Unpack 4 values per byte
    values = []
    for byte in data:
        for shift in [0, 2, 4, 6]:
            nb = (byte >> shift) & 3
            values.append(ternary_lut[nb])
    
    return np.array(values, dtype=np.float32)

def quantize_float_to_4bit(float_values):
    """Quantize float32 to 4-bit signed (16 levels: -7.5 to 7.5)."""
    if len(float_values) == 0:
        return np.array([], dtype=np.uint8)
    
    # Clip to range
    clipped = np.clip(float_values, -7.5, 7.5)
    
    # Map to 0-15: value + 7.5 -> round -> clip
    q4 = np.round(clipped + 7.5).astype(np.uint8)
    q4 = np.clip(q4, 0, 15)
    
    # Pack 2 values per byte
    packed = np.zeros(len(q4) // 2, dtype=np.uint8)
    for i in range(0, len(q4), 2):
        packed[i//2] = (q4[i] << 4) | q4[i+1]
    
    return packed

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_convert_4bit_chunked.py <input_2bit.bqsm> <output_4bit.bqsm>")
        sys.exit(1)
    
    success = convert_chunked(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)

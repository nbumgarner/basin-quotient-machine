#!/usr/bin/env python3
"""
Convert 2-bit ternary BQSM to 4-bit BQSM for better coherence.
4-bit: 16 levels (-7.5 to 7.5) vs 2-bit ternary (-1, 0, 1)
"""
import struct
import numpy as np
import sys
import os

# 4-bit signed scale: -7.5 to 7.5 in steps of 1
# Values: -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5
# Map to 4-bit: 0=-7.5, 1=-6.5, ..., 15=7.5
# Or simpler: use int4 with scale factor

def read_bqsm_header(filepath):
    """Read BQSM file header."""
    with open(filepath, 'rb') as f:
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
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        
        return {
            'version': version,
            'd_model': d_model,
            'ffn_dim': ffn_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'vocab_size': vocab_size,
            'n_tensors': n_tensors
        }

def dequantize_2bit_to_float(packed_bytes, scale=1.0):
    """Dequantize 2-bit ternary to float32."""
    # Ternary LUT: 0=-1, 1=0, 2=1, 3=0 (for signed)
    # Actually: 0=-1, 1=0, 2=1, 3=0 (from bqsm_ternary_lut)
    ternary_lut = np.array([-1, 0, 1, 0], dtype=np.float32)
    
    values = []
    for byte in packed_bytes:
        for shift in [0, 2, 4, 6]:
            nb = (byte >> shift) & 3
            values.append(ternary_lut[nb] * scale)
    
    return np.array(values, dtype=np.float32)

def quantize_float_to_4bit(float_values):
    """Quantize float32 to 4-bit signed."""
    # 4-bit signed: 16 levels from -7.5 to 7.5
    # Map: value -> round(value + 7.5) -> clip to [0, 15]
    q4_values = np.round(float_values + 7.5).astype(np.uint8)
    q4_values = np.clip(q4_values, 0, 15)
    
    # Pack 2 values per byte
    packed = np.zeros(len(q4_values) // 2, dtype=np.uint8)
    for i in range(0, len(q4_values), 2):
        packed[i//2] = (q4_values[i] << 4) | q4_values[i+1]
    
    return packed

def convert_bqsm_2bit_to_4bit(input_path, output_path):
    """Convert 2-bit BQSM to 4-bit BQSM."""
    print(f"Reading 2-bit BQSM: {input_path}")
    
    # Read header
    header = read_bqsm_header(input_path)
    print(f"  d_model={header['d_model']}, layers={header['n_layers']}, vocab={header['vocab_size']}")
    
    # Read tensors
    tensors = []
    with open(input_path, 'rb') as f:
        # Skip header (44 bytes)
        f.read(44)
        
        for i in range(header['n_tensors']):
            # Read tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            # Read tensor type
            tensor_type = struct.unpack('<I', f.read(4))[0]
            
            # Read shape
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(ndims):
                dim = struct.unpack('<Q', f.read(8))[0]
                shape.append(dim)
            
            # Read data length and data
            data_len = struct.unpack('<Q', f.read(8))[0]
            data = f.read(data_len)
            
            tensors.append({
                'name': name,
                'type': tensor_type,
                'shape': shape,
                'data': data,
                'data_len': data_len
            })
            
            if i % 50 == 0:
                print(f"  Read {i}/{header['n_tensors']} tensors...")
    
    print(f"Converting {len(tensors)} tensors to 4-bit...")
    
    # Convert to 4-bit BQSM
    bqsm_data = struct.pack('<4sIIIIIIIQ',
        b'BQSM',  # magic
        2,        # version (4-bit)
        header['d_model'],
        header['ffn_dim'],
        header['n_layers'],
        header['n_heads'],
        header['n_kv_heads'],
        header['vocab_size'],
        len(tensors)
    )
    
    for t in tensors:
        # Dequantize 2-bit to float
        if len(t['data']) > 0:
            float_values = dequantize_2bit_to_float(t['data'])
            
            # Quantize to 4-bit
            packed_4bit = quantize_float_to_4bit(float_values)
        else:
            packed_4bit = t['data']
        
        # Store tensor info
        name_bytes = t['name'].encode('utf-8')
        bqsm_data += struct.pack('<Q', len(name_bytes))
        bqsm_data += name_bytes
        bqsm_data += struct.pack('<I', t['type'])
        bqsm_data += struct.pack('<I', len(t['shape']))
        for dim in t['shape']:
            bqsm_data += struct.pack('<Q', dim)
        bqsm_data += struct.pack('<Q', len(packed_4bit))
        bqsm_data += packed_4bit.tobytes()
        
        if (tensors.index(t) + 1) % 50 == 0:
            print(f"  Converted {(tensors.index(t) + 1)}/{len(tensors)} tensors...")
    
    # Write output
    print(f"Writing 4-bit BQSM: {output_path} ({len(bqsm_data)} bytes)")
    with open(output_path, 'wb') as f:
        f.write(bqsm_data)
    
    print(f"Done! Output: {output_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_convert_4bit.py <input_2bit.bqsm> <output_4bit.bqsm>")
        sys.exit(1)
    
    success = convert_bqsm_2bit_to_4bit(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Convert 2-bit ternary BQSM to 4-bit BQSM using PROPER dequantization.
Uses the same approach as bqsm_convert_real.py for accurate weight recovery.
"""
import struct
import numpy as np
import sys
import os

def read_bqsm_2bit(filepath):
    """Read 2-bit BQSM file and return tensors as float32."""
    print(f"Reading 2-bit BQSM: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read header
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
        
        print(f"  d_model={d_model}, layers={n_layers}, vocab={vocab_size}")
        
        # Read tensors
        tensors = []
        for i in range(n_tensors):
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
            
            # Read data
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
                print(f"  Read {i}/{n_tensors} tensors...")
    
    return {
        'version': version,
        'd_model': d_model,
        'ffn_dim': ffn_dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'vocab_size': vocab_size,
        'tensors': tensors
    }

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

def write_bqsm_4bit(output_path, header, tensors):
    """Write 4-bit BQSM file."""
    print(f"Writing 4-bit BQSM: {output_path}")
    
    bqsm_data = struct.pack('<4sIIIIIIIQ',
        b'BQSM',  # magic
        4,        # version (4-bit)
        header['d_model'],
        header['ffn_dim'],
        header['n_layers'],
        header['n_heads'],
        header['n_kv_heads'],
        header['vocab_size'],
        len(tensors)
    )
    
    for t in tensors:
        name_bytes = t['name'].encode('utf-8')
        bqsm_data += struct.pack('<Q', len(name_bytes))
        bqsm_data += name_bytes
        bqsm_data += struct.pack('<I', t['type'])
        bqsm_data += struct.pack('<I', len(t['shape']))
        for dim in t['shape']:
            bqsm_data += struct.pack('<Q', dim)
        bqsm_data += struct.pack('<Q', len(t['data']))
        bqsm_data += t['data']
        
        if (tensors.index(t) + 1) % 50 == 0:
            print(f"  Wrote {(tensors.index(t) + 1)}/{len(tensors)} tensors...")
    
    with open(output_path, 'wb') as f:
        f.write(bqsm_data)
    
    print(f"Done! Size: {len(bqsm_data)} bytes")
    return len(bqsm_data)

def convert_2bit_to_4bit(input_path, output_path):
    """Convert 2-bit BQSM to 4-bit BQSM."""
    # Read 2-bit file
    data = read_bqsm_2bit(input_path)
    header = {
        'version': data['version'],
        'd_model': data['d_model'],
        'ffn_dim': data['ffn_dim'],
        'n_layers': data['n_layers'],
        'n_heads': data['n_heads'],
        'n_kv_heads': data['n_kv_heads'],
        'vocab_size': data['vocab_size']
    }
    
    print(f"Converting {len(data['tensors'])} tensors to 4-bit...")
    
    # Convert each tensor
    converted_tensors = []
    for i, t in enumerate(data['tensors']):
        # Dequantize 2-bit to float
        float_values = dequantize_2bit_to_float(t['data'])
        
        # Quantize to 4-bit
        packed_4bit = quantize_float_to_4bit(float_values)
        
        converted_tensors.append({
            'name': t['name'],
            'type': t['type'],
            'shape': t['shape'],
            'data': packed_4bit.tobytes()
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Converted {i+1}/{len(data['tensors'])} tensors...")
        
        # Free memory
        del float_values
    
    # Write 4-bit file
    size = write_bqsm_4bit(output_path, header, converted_tensors)
    
    print(f"\nConversion complete!")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Size: {size} bytes")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: bqsm_convert_4bit_proper.py <input_2bit.bqsm> <output_4bit.bqsm>")
        sys.exit(1)
    
    success = convert_2bit_to_4bit(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)

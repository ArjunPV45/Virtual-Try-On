import struct
import json

def load_and_print_glb(glb_file):
    print("=" * 60)
    print(f"Loading: {glb_file}")
    print("=" * 60)
    
    with open(glb_file, 'rb') as f:
        file_data = f.read()
    
    print(f"✓ File loaded: {len(file_data):,} bytes\n")
    
    print("─" * 60)
    print("HEADER (First 12 bytes)")
    print("─" * 60)
    
    magic, version, length = struct.unpack('<4sII', file_data[:12])
    print(f"Magic: {magic.decode('utf-8')}")
    print(f"Version: {version}")
    print(f"File length: {length:,} bytes\n")
    
    print("─" * 60)
    print("JSON CHUNK")
    print("─" * 60)
    
    json_size, json_type = struct.unpack('<I4s', file_data[12:20])
    print(f"JSON size: {json_size:,} bytes\n")
    
    json_text = file_data[20:20+json_size].decode('utf-8')
    json_dict = json.loads(json_text)
    
    print("─" * 60)
    print("CONTENTS SUMMARY")
    print("─" * 60)
    
    print(f"Meshes: {len(json_dict.get('meshes', []))}")
    print(f"Materials: {len(json_dict.get('materials', []))}")
    print(f"Textures: {len(json_dict.get('textures', []))}")
    print(f"Images: {len(json_dict.get('images', []))}")
    print(f"Nodes: {len(json_dict.get('nodes', []))}\n")
    
    
    print("─" * 60)
    print("BINARY DATA")
    print("─" * 60)
    
    binary_chunk_start = 20 + json_size
    binary_size = struct.unpack('<I', file_data[binary_chunk_start:binary_chunk_start+4])[0]
    binary_start = binary_chunk_start + 8
    binary_data = file_data[binary_start:binary_start+binary_size]
    
    print(f"Binary data size: {binary_size:,} bytes\n")
    
    print("─" * 60)
    print("MESH DETAILS")
    print("─" * 60)
    
    if json_dict.get('meshes'):
        mesh = json_dict['meshes'][0]
        print(f"First mesh: {mesh.get('name', 'Unnamed')}")
        
        primitive = mesh['primitives'][0]
        print(f"Primitives: {len(mesh['primitives'])}")
        
        pos_accessor_idx = primitive['attributes']['POSITION']
        pos_accessor = json_dict['accessors'][pos_accessor_idx]
        vertex_count = pos_accessor['count']
        print(f"Vertices: {vertex_count:,}\n")
        
        indices_accessor_idx = primitive['indices']
        indices_accessor = json_dict['accessors'][indices_accessor_idx]
        index_count = indices_accessor['count']
        print(f"Indices: {index_count:,}")
        print(f"Triangles: {index_count // 3:,}\n")
        
        print("─" * 60)
        print("FIRST 3 VERTICES")
        print("─" * 60)
        
        buffer_view = json_dict['bufferViews'][pos_accessor['bufferView']]
        offset = buffer_view.get('byteOffset', 0)
        
        for i in range(3):
            pos = offset + (i * 12)
            x, y, z = struct.unpack_from('fff', binary_data, pos)
            print(f"Vertex {i}: ({x:.4f}, {y:.4f}, {z:.4f})")
        
        print()
    
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


load_and_print_glb("models/glasses.glb")

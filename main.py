import cv2
import struct
import mediapipe as mp
import numpy as np
import json

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

def load_glb_data(glb_file):
    with open(glb_file, 'rb') as f:
        file_data = f.read()

    json_size = struct.unpack('<I', file_data[12:16])[0]
    json_text = file_data[20:20+json_size].decode('utf-8')
    json_data = json.loads(json_text)

    binary_chunk_start = 20 + json_size
    binary_size = struct.unpack('<I', file_data[binary_chunk_start:binary_chunk_start+4])[0]
    binary_start = binary_chunk_start + 8
    binary_data = file_data[binary_start:binary_start+binary_size]

    mesh = json_data['meshes'][0]
    primitive = mesh['primitives'][0]

    pos_accessor_idx = primitive['attributes']['POSITION']
    pos_accessor = json_data['accessors'][pos_accessor_idx]
    buffer_view = json_data['bufferViews'][pos_accessor['bufferView']]

    vertex_count = pos_accessor['count']
    byte_offset = buffer_view.get('byteOffset', 0)

    vertices = []
    for i in range(vertex_count):
        pos = byte_offset + (i * 12)
        x, y, z = struct.unpack_from('fff', binary_data, pos)
        vertices.append([x, y, z])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    print(f"✓ Loaded {len(vertices)} vertices")
    print(f"  Sample vertex 0: {vertices[0]}")
    print(f"  Sample vertex 1: {vertices[1]}")
    
    return vertices

def transform_vertices_with_pose(vertices, rotation_vector, translation_vector, scale=1.0):
    """Transform vertices using head pose"""
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print("Rotation Matrix (3x3):")
    print(rotation_matrix)
    print()

    transformed = []

    for i, vertex in enumerate(vertices):
        rotated = rotation_matrix @ (vertex * scale)
        transformed_vertex = rotated + translation_vector.flatten()[:3]
        transformed.append(transformed_vertex)

        if i < 5:
            print(f"Vertex {i}:")
            print(f"  Original: {vertex}")
            print(f"  Transformed: {transformed_vertex}")

    transformed = np.array(transformed, dtype=np.float32)
    print(f"\n✓ Transformed {len(transformed)} vertices\n")

    return transformed

class HeadPoseEstimator:
    """Estimate 3D head pose from 2D landmarks"""
    
    def __init__(self, frame_width, frame_height):
        focal_length = frame_width
        self.camera_matrix = np.array([
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        print("Camera Matrix:")
        print(self.camera_matrix)
        print()

        self.dist_coeffs = np.zeros((4, 1))

        
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           
            (0.0, -330.0, -65.0),      
            (-225.0, 170.0, -135.0),   
            (225.0, 170.0, -135.0),    
            (-150.0, -150.0, -125.0),  
            (150.0, -150.0, -125.0)    
        ], dtype=np.float64)

        print("3D Face Model Points (mm):")
        for i, pt in enumerate(self.model_points):
            print(f"  Point {i}: {pt}")
        print()

    def estimate_pose(self, landmarks, frame_width, frame_height):
        
        pose_landmark_indices = [1, 152, 33, 263, 61, 291]

        image_points = []
        for idx in pose_landmark_indices:
            lm = landmarks[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            image_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float64)

        print("2D Image Points (pixels):")
        for i, pt in enumerate(image_points):
            print(f"  Point {i}: {pt}")
        print()

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            print("✓ Pose estimated successfully")
            print(f"  Rotation Vector: {rotation_vector.flatten()}")
            print(f"  Translation Vector: {translation_vector.flatten()}")
            print()
        else:
            print("✗ Pose Estimation Failed")
        
        return success, rotation_vector, translation_vector

if __name__ == "__main__":
    print("\n" + "="*60)
    print("3D VIRTUAL GLASSES - STEP 1-5")
    print("="*60 + "\n")
    
    print("Loading glasses model...")
    glasses_vertices = load_glb_data("models/glasses.glb")
    print()

    print("Testing with dummy pose...")
    rotation_vec = np.array([0.1, -0.05, 0.02])
    translation_vec = np.array([5.2, -10.3, 250.1])

    transformed_vertices = transform_vertices_with_pose(
        glasses_vertices,
        rotation_vec,
        translation_vec,
        scale=1.0
    )

    print("Initializing camera and pose estimator...")
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_estimator = HeadPoseEstimator(frame_width, frame_height)
    
    print("="*60)
    print("Ready! Move your head and watch the console output")
    print("Press 'q' to quit")
    print("="*60 + "\n")

    frame_count = 0

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            pos_success, rotation_vec, translation_vec = pose_estimator.estimate_pose(
                landmarks, frame_width, frame_height
            )

            if pos_success:
                transformed_vertices = transform_vertices_with_pose(
                    glasses_vertices,
                    rotation_vec,
                    translation_vec,
                    scale=1.0
                )

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"\n--- Frame {frame_count} ---\n")
            
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("\n✓ Done!")

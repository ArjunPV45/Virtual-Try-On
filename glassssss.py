import cv2
import struct
import mediapipe as mp
import numpy as np
import json
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

# ============= OPENGL RENDERER CLASS =============
class SimpleGlassesRenderer:
    
    def __init__(self, vertices, faces, width=1280, height=720):
        
        self.width = width
        self.height = height
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.uint32)
        
        print("\nInitializing OpenGL...")
        
        # Initialize pygame
        pygame.init()
        
        # Create hidden window (important for Raspberry Pi)
        try:
            self.display = pygame.display.set_mode(
                (width, height), 
                DOUBLEBUF | OPENGL | pygame.HIDDEN
            )
        except:
            # Fallback if HIDDEN not supported
            self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        
        pygame.display.set_caption("Glasses Renderer")
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)              # Enable depth testing (3D)
        glEnable(GL_BLEND)                   # Enable transparency
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 0)             # Transparent background
        
        # Set viewport and projection
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (width / height), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        
        print("✓ OpenGL context created")
        
        # Upload mesh to GPU
        self._upload_mesh_to_gpu()
        
        print("✓ Renderer initialized\n")
    
    def _upload_mesh_to_gpu(self):
        """Upload vertices and faces to GPU memory"""
        
        print("Uploading mesh to GPU...")
        
        # Create Vertex Array Object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create Vertex Buffer Object (VBO) for vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER, 
            self.vertices.nbytes, 
            self.vertices, 
            GL_STATIC_DRAW
        )
        
        # Create Element Buffer Object (EBO) for faces/indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, 
            self.faces.nbytes, 
            self.faces, 
            GL_STATIC_DRAW
        )
        
        # Set vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Calculate total indices
        self.face_count = len(self.faces) * 3
        
        glBindVertexArray(0)
        
        print(f"✓ Uploaded {len(self.vertices):,} vertices")
        print(f"✓ Uploaded {len(self.faces):,} faces ({self.face_count:,} indices)\n")
    
    def render_to_image(self, rotation_vec, translation_vec, scale=1.0):
        """
        Render glasses to an image
        
        Input:
          - rotation_vec: (3,) rotation from solvePnP
          - translation_vec: (3,) translation from solvePnP
          - scale: size multiplier
        
        Output:
          - RGBA image (1280x720x4)
        """
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Move camera back
        glTranslatef(0, 0, -150)
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        
        # Build 4x4 transformation matrix
        # This combines rotation, translation, and scale
        gl_matrix = np.eye(4, dtype=np.float32)
        gl_matrix[:3, :3] = rotation_matrix * scale
        gl_matrix[:3, 3] = translation_vec.flatten()[:3]
        
        # Apply transformation
        glMultMatrixf(gl_matrix.T.flatten())
        
        # Set color: Dark gray, slightly transparent
        glColor4f(0.2, 0.2, 0.2, 0.95)
        
        # Draw the mesh
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.face_count, GL_UNSIGNED_INT, None)
        
        # Read pixels from OpenGL back buffer
        glReadBuffer(GL_BACK)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and reshape
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
        
        # Flip Y axis (OpenGL origin is bottom-left, OpenCV is top-left)
        image = np.flipud(image)
        
        return image
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteVertexArrays(1, [self.vao])
        pygame.quit()
        print("✓ Cleaned up OpenGL")


# ============= COMPOSITOR =============
def composite_rgba_onto_bgr(bgr_frame, rgba_glasses):
    """
    Blend RGBA glasses onto BGR webcam frame
    
    Formula: Output = (alpha * glasses_color) + ((1-alpha) * frame_color)
    """
    
    h_bg, w_bg = bgr_frame.shape[:2]
    h_gl, w_gl = rgba_glasses.shape[:2]
    
    # Resize if needed
    if (w_gl, h_gl) != (w_bg, h_bg):
        rgba_glasses = cv2.resize(rgba_glasses, (w_bg, h_bg))
    
    # Extract alpha channel (0-1 range)
    alpha = rgba_glasses[:, :, 3:4].astype(float) / 255.0
    
    # Extract RGB from glasses (0-255 range)
    glasses_rgb = rgba_glasses[:, :, :3].astype(float)
    
    # Convert BGR to RGB for proper blending
    frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(float)
    
    # Blend
    composite_rgb = alpha * glasses_rgb + (1 - alpha) * frame_rgb
    composite_rgb = np.clip(composite_rgb, 0, 255).astype(np.uint8)
    
    # Convert back to BGR
    composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
    
    return composite_bgr


# ============= MAIN APPLICATION =============
class VirtualGlassesApp:
    """Main app combining everything"""
    
    def __init__(self, glb_path='models/glasses.glb'):
        """Initialize app"""
        
        print("\n" + "="*70)
        print("3D VIRTUAL GLASSES TRY-ON - STEP 6: OPENGL RENDERING")
        print("="*70)
        
        # Load mesh
        print("\n1. Loading 3D model...")
        self.vertices, self.faces = self._load_glb(glb_path)
        
        # Initialize camera
        print("\n2. Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"   Camera: {self.frame_width}x{self.frame_height}")
        
        # Initialize renderer
        print("\n3. Initializing OpenGL renderer...")
        self.renderer = SimpleGlassesRenderer(
            self.vertices, 
            self.faces,
            width=self.frame_width,
            height=self.frame_height
        )
        
        # Initialize pose estimator
        print("4. Initializing pose estimator...")
        self.pose_estimator = self._init_pose_estimator()
        
        # Initialize face mesh
        print("5. Initializing face detection...")
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("\n" + "="*70)
        print("✓ All systems ready! Starting virtual try-on...")
        print("="*70 + "\n")
    
    def _load_glb(self, glb_path):
        """Load vertices and faces from GLB"""
        
        with open(glb_path, 'rb') as f:
            file_data = f.read()
        
        # Parse JSON
        json_size = struct.unpack('<I', file_data[12:16])[0]
        json_text = file_data[20:20+json_size].decode('utf-8')
        json_data = json.loads(json_text)
        
        # Extract binary
        binary_chunk_start = 20 + json_size
        binary_size = struct.unpack('<I', file_data[binary_chunk_start:binary_chunk_start+4])[0]
        binary_start = binary_chunk_start + 8
        binary_data = file_data[binary_start:binary_start+binary_size]
        
        # Extract mesh
        mesh = json_data['meshes'][0]
        primitive = mesh['primitives'][0]
        
        # Extract vertices
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
        
        # Normalize
        vertices -= vertices.mean(axis=0)
        max_dist = np.max(np.linalg.norm(vertices, axis=1))
        vertices = vertices / max_dist * 50
        
        print(f"   ✓ Loaded {len(vertices):,} vertices")
        
        # Extract faces
        indices_accessor_idx = primitive['indices']
        indices_accessor = json_data['accessors'][indices_accessor_idx]
        buffer_view = json_data['bufferViews'][indices_accessor['bufferView']]
        
        index_count = indices_accessor['count']
        byte_offset = buffer_view.get('byteOffset', 0)
        
        indices = []
        for i in range(index_count):
            pos = byte_offset + (i * 4)
            idx = struct.unpack_from('I', binary_data, pos)[0]
            indices.append(idx)
        
        faces = np.array(indices, dtype=np.uint32).reshape(-1, 3)
        print(f"   ✓ Loaded {len(faces):,} faces")
        
        return vertices, faces
    
    def _init_pose_estimator(self):
        """Initialize pose estimation"""
        
        focal_length = self.frame_width
        camera_matrix = np.array([
            [focal_length, 0, self.frame_width / 2],
            [0, focal_length, self.frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        
        print("   ✓ Pose estimator ready")
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'model_points': model_points
        }
    
    def _estimate_pose(self, landmarks):
        """Estimate head pose from landmarks"""
        
        pose_landmark_indices = [1, 152, 33, 263, 61, 291]
        
        image_points = np.array([
            [int(landmarks[idx].x * self.frame_width),
             int(landmarks[idx].y * self.frame_height)]
            for idx in pose_landmark_indices
        ], dtype=np.float64)
        
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.pose_estimator['model_points'],
            image_points,
            self.pose_estimator['camera_matrix'],
            self.pose_estimator['dist_coeffs'],
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return success, rotation_vec, translation_vec
    
    def run(self):
        """Main loop"""
        
        print("Press 'q' to quit\n")
        
        frame_count = 0
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    continue
                
                # Detect face
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Estimate pose
                    pose_success, rotation_vec, translation_vec = self._estimate_pose(landmarks)
                    
                    if pose_success:
                        # Render 3D glasses
                        glasses_rgba = self.renderer.render_to_image(
                            rotation_vec, 
                            translation_vec,
                            scale=1.2  # Adjust for size
                        )
                        
                        # Composite onto frame
                        frame = composite_rgba_onto_bgr(frame, glasses_rgba)
                        
                        # Status
                        cv2.putText(frame, "3D Glasses Detected", (20, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Pose Failed", (20, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "No Face", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
                
                # Display
                cv2.imshow('3D Virtual Glasses Try-On', cv2.flip(frame, 1))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    break
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.renderer.cleanup()
        print("✓ Done!")


# ============= MAIN =============
import ctypes

if __name__ == "__main__":
    try:
        app = VirtualGlassesApp('models/glasses.glb')
        app.run()
    except Exception as e:
        print(f"Failed to start: {e}")
        import traceback
        traceback.print_exc()

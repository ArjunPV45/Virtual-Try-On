import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
cam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("No camera frame")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        LANDMARKS_FOR_GLASSES = [159, 145, 386, 374, 127, 356] #[33, 144, 160, 153, 158, 133]


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=drawing_spec, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                #mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                #mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=drawing_spec, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                #print('face_landmarks:', face_landmarks)

                for idx in LANDMARKS_FOR_GLASSES:
                    mark = face_landmarks.landmark[idx]
                    print("mark", mark)
                    x, y = int(mark.x * image_width), int(mark.y * image_height)
                    cv2.circle(image, (x, y), 4, (0, 255, 255), -1)
                    cv2.putText(image, str(idx), (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

                '''left_ear_landmarks = [127, 234, 162, 356]
                right_ear_landmarks = [356, 454, 389, 390]

                for idx in left_ear_landmarks:
                    landmark = face_landmarks.landmark[idx]
                    px = int(landmark.x * image_width)
                    py = int(landmark.y * image_height)
                    cv2.circle(image, (px, py), 5, (255, 0, 0), -1)

                for idx in right_ear_landmarks:
                    landmark = face_landmarks.landmark[idx]
                    px = int(landmark.x * image_width)
                    py = int(landmark.y * image_height)
                    cv2.circle(image, (px, py), 5, (255, 0, 0), -1)'''
                


        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(1) == ord('q'):
            break

cam.release()





import cv2
import os
import numpy as np
import mediapipe as mp

class FaceFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_dict = {
                    "right_eye": (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y),
                    "left_eye": (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y),
                    "nose": (face_landmarks.landmark[1].x, face_landmarks.landmark[1].y),
                    "mouth_right": (face_landmarks.landmark[61].x, face_landmarks.landmark[61].y),
                    "mouth_left": (face_landmarks.landmark[291].x, face_landmarks.landmark[291].y)
                }
                landmarks_list.append(landmarks_dict)
        return landmarks_list

    def draw_landmarks(self, image, landmarks_list):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        for landmarks in landmarks_list:
            for key, (x, y) in landmarks.items():
                x, y = int(x * width), int(y * height)
                cv2.circle(image_rgb, (x, y), 2, (0, 255, 0), -1)
        return image_rgb

class PoseClassifier:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def is_frontal_face(self, landmarks):
        if landmarks:
            right_eye = np.array(landmarks["right_eye"])
            left_eye = np.array(landmarks["left_eye"])
            nose = np.array(landmarks["nose"])
            mouth_right = np.array(landmarks["mouth_right"])
            mouth_left = np.array(landmarks["mouth_left"])

            eye_distance = np.linalg.norm(right_eye - left_eye)
            nose_to_right_eye = np.linalg.norm(nose - right_eye)
            nose_to_left_eye = np.linalg.norm(nose - left_eye)
            mouth_distance = np.linalg.norm(mouth_right - mouth_left)
            nose_to_mouth_right = np.linalg.norm(nose - mouth_right)
            nose_to_mouth_left = np.linalg.norm(nose - mouth_left)

            if (abs(nose_to_right_eye - nose_to_left_eye) < eye_distance * 0.14 and
                abs(nose_to_mouth_right - nose_to_mouth_left) < mouth_distance * 0.14):
                return True
        return False

def process_and_filter_faces(input_dir, output_dir, feature_extractor, pose_classifier):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir)]

    for file in image_files:
        img_path = os.path.join(input_dir, file)
        image = cv2.imread(img_path)

        # Extract facial landmarks
        landmarks_list = feature_extractor.extract_landmarks(image)

        if landmarks_list:
            for i, landmarks in enumerate(landmarks_list):
                # Check if the face is in a frontal pose
                if pose_classifier.is_frontal_face(landmarks):
                    # Crop the face
                    x1, y1, x2, y2 = (int(landmarks["right_eye"][0] * image.shape[1] - 50),
                                      int(landmarks["right_eye"][1] * image.shape[0] - 50),
                                      int(landmarks["mouth_left"][0] * image.shape[1] + 50),
                                      int(landmarks["mouth_left"][1] * image.shape[0] + 50))
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, image.shape[1]), min(y2, image.shape[0])
                    frontal_face = image[y1:y2, x1:x2]

                    # Save the cropped frontal face
                    frontal_face_filename = f"{os.path.splitext(file)[0]}_frontal_face_{i}.jpg"
                    frontal_face_path = os.path.join(output_dir, frontal_face_filename)
                    cv2.imwrite(frontal_face_path, frontal_face)
                    print(f"Saved cropped frontal face to {frontal_face_path}")
        else:
            print(f"No landmarks detected in {file}, skipping.")

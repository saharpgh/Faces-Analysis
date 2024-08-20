import cv2
from retinaface import RetinaFace
import os

class FaceDetectorRetinaFace:
    def __init__(self):
        self.detector = RetinaFace

    def detect_faces(self, image_path):
        # Detect faces in the image
        faces = self.detector.detect_faces(image_path)
        return faces

    def crop_faces(self, image, faces):
        cropped_faces = []
        for key in faces.keys():
            identity = faces[key]
            facial_area = identity["facial_area"]
            x1, y1, x2, y2 = facial_area
            cropped_face = image[y1:y2, x1:x2]
            cropped_faces.append(cropped_face)
        return cropped_faces

def process_images_and_save(dataset_path, output_path, face_detector):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all files in the dataset directory
    image_files = [f for f in os.listdir(dataset_path)]

    for file in image_files:
        img_path = os.path.join(dataset_path, file)
        image = cv2.imread(img_path)

        # Detect faces
        faces = face_detector.detect_faces(img_path)

        if faces:
            # Crop faces from the image
            cropped_faces = face_detector.crop_faces(image, faces)

            # Save each cropped face
            for i, cropped_face in enumerate(cropped_faces):
                face_file_name = f"{os.path.splitext(file)[0]}_face_{i}.jpg"
                face_file_path = os.path.join(output_path, face_file_name)
                cv2.imwrite(face_file_path, cropped_face)
                print(f"Saved cropped face to {face_file_path}")
        else:
            print(f"No faces detected in {file}, skipping.")

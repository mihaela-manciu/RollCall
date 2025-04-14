import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import time
start_time=time.time()

vgg16_model = load_model('D:/Poli/Licenta/RollCall/RollCall/TestCNN/test_VGG16_0.0001_50.model.keras')
embeddings = pd.read_csv('D:/Poli/Licenta/RollCall/RollCall/TestPrototypical/class_embeddings_50_5_2_1.csv', index_col=0).values

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def recognize_faces(image, faces, model, embeddings, class_names):
    recognized_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0) / 255.0
        face_features = model.predict(face)
        distances = np.linalg.norm(embeddings - face_features, axis=1)
        min_distance = np.min(distances)
        if min_distance < 0.5:
            class_name = class_names[np.argmin(distances)]
        else:
            class_name = 'Unknown'
        recognized_faces.append((x, y, w, h, class_name))
    return recognized_faces

def main():
    image_path = 'D:/Poli/Licenta/RollCall/RollCall/Media/Group_Photos/group_20.jpg'
    image = cv2.imread(image_path)
    faces = detect_faces(image)
    class_names = pd.read_csv('D:/Poli/Licenta/RollCall/RollCall/TestPrototypical/class_embeddings_50_5_2_1.csv', index_col=0).index.to_list()
    recognized_faces = recognize_faces(image, faces, vgg16_model, embeddings, class_names)

    for (x, y, w, h, class_name) in recognized_faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    screen_width = 1300
    screen_height = 1000
    resized_image = cv2.resize(image, (screen_width, screen_height))
    cv2.imshow('Recognized Faces', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Number of recognized faces: {len(recognized_faces)}')
    for _, _, _, _, class_name in recognized_faces:
        print(f'Recognized: {class_name}')

    end_time = time.time()
    total_time_seconds = end_time - start_time
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    print("Model compilation time:", minutes, "minutes", seconds, "seconds")
if __name__ == '__main__':
    main()

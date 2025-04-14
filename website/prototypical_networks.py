from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import cv2
import time

start_time=time.time()


vgg16_model = load_model('D:/Poli/Licenta/RollCall/RollCall/TestCNN/test_VGG16_0.0001_50.model.keras')


def preprocess_image(image_path):
    target_size = (224,224)
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def generate_episode(data, num_classes, num_support, num_query):
    selected_classes = np.random.choice(list(data.keys()), num_classes, replace=False)
    support_set = {cls: random.sample(data[cls], num_support) for cls in selected_classes}
    query_set = {cls: random.sample(data[cls], num_query) for cls in selected_classes}
    return support_set, query_set


def compute_prototypes(support_set, vgg16_model):
    prototypes = {}
    for cls, images in support_set.items():
        embeddings = [vgg16_model.predict(img[np.newaxis, ...])[0] for img in images]
        prototypes[cls] = np.mean(embeddings, axis=0)
    return prototypes


def classify_and_compute_loss(query_set, prototypes, vgg16_model):
    total_loss = 0
    correct = 0
    total = 0
    correct_classes = []
    for cls, images in query_set.items():
        for img in images:
            embedding = vgg16_model.predict(img[np.newaxis, ...])[0]
            distances = {proto_cls: np.linalg.norm(embedding - proto) for proto_cls, proto in prototypes.items()}
            predicted_cls = min(distances, key=distances.get)
            if predicted_cls == cls:
                correct += 1
                correct_classes.append(cls)
            total += 1
            total_loss += distances[cls]
    accuracy = correct / total
    loss = total_loss / total
    return accuracy, loss, correct_classes


data_dir = 'D:/Poli/Licenta/RollCall/RollCall/Media/People'  # replace with the path to your dataset
classes = os.listdir(data_dir)
data = {
    cls: [preprocess_image(os.path.join(data_dir, cls, img)) for img in os.listdir(os.path.join(data_dir, cls))
          if img.endswith(('jpg', 'png'))] for cls in classes}
correct_classes_per_episode = []

num_episodes = 25 #....100?
num_classes_per_episode = 5 #...10
num_support_per_class = 2 #...4
num_query_per_class = 1

accuracies = []
losses = []


for episode in range(num_episodes):
    support_set, query_set = generate_episode(data, num_classes_per_episode, num_support_per_class, num_query_per_class)
    prototypes = compute_prototypes(support_set, vgg16_model)
    accuracy, loss, correct_classes = classify_and_compute_loss(query_set, prototypes, vgg16_model)
    accuracies.append(accuracy)
    losses.append(loss)
    correct_classes_per_episode.append(correct_classes)
    print(f'Episode {episode}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')
    print(f'Support Set Classes: {list(support_set.keys())}')
    print(f'Query Set Classes: {list(query_set.keys())}')
    print(f'Correctly Guessed Classes: {correct_classes}')

    end_time = time.time()
    total_time_seconds = end_time - start_time
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    print("Model compilation time:", minutes, "minutes", seconds, "seconds")

prototypes = compute_prototypes({cls: data[cls][:num_support_per_class] for cls in classes}, vgg16_model)
embeddings = {cls: proto.tolist() for cls, proto in prototypes.items()}
df = pd.DataFrame.from_dict(embeddings, orient='index')
df.to_csv('D:/Poli/Licenta/RollCall/RollCall/TestPrototypical/class_embeddings_25_5_2_1.csv', header=False)


for i in range(2):
    support_set, query_set = generate_episode(data, num_classes_per_episode, num_support_per_class, num_query_per_class)
    plt.figure(figsize=(10, 5))
    print(f'Support Set Classes (Episode {i}): {list(support_set.keys())}')
    print(f'Query Set Classes (Episode {i}): {list(query_set.keys())}')
    for j, (cls, images) in enumerate(support_set.items()):
        for k, img in enumerate(images):
            plt.subplot(num_classes_per_episode, num_support_per_class, j * num_support_per_class + k + 1)
            plt.imshow(img)
            # plt.title(f'Support: {cls}')
            plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 2))
    for j, (cls, images) in enumerate(query_set.items()):
        for k, img in enumerate(images):
            plt.subplot(num_classes_per_episode, num_query_per_class, j * num_query_per_class + k + 1)
            plt.imshow(img)
            # plt.title(f'Query: {cls}')
            plt.axis('off')
    plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_episodes), accuracies)
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.title('Accuracy over Episodes')

plt.subplot(1, 2, 2)
plt.plot(range(num_episodes), losses)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss over Episodes')

plt.tight_layout()
plt.show()

average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy:.4f}')
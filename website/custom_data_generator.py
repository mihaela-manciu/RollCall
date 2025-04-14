

#### KEEP ####


import os
import numpy as np
import keras
from keras.utils import Sequence
import random
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NumpyDataGenerator(Sequence):
    def __init__(self, directory, batch_size=16, target_size=(224, 224), shuffle=True, augment=False,
                 validation_split=0.2, subset=None):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.validation_split = validation_split
        self.subset = subset

        self.filepaths, self.labels = self._load_filepaths_and_labels()
        self.train_filepaths, self.val_filepaths, self.train_labels, self.val_labels = train_test_split(
            self.filepaths, self.labels, test_size=self.validation_split, stratify=self.labels
        )

        if self.subset == 'training':
            self.filepaths = self.train_filepaths
            self.labels = self.train_labels
        elif self.subset == 'validation':
            self.filepaths = self.val_filepaths
            self.labels = self.val_labels

        self.on_epoch_end()

        if self.augment:
            self.augmentation = ImageDataGenerator(
                zoom_range=0.2,
                rotation_range=20,
                fill_mode='nearest'
            )
        else:
            self.augmentation = ImageDataGenerator()

    def _load_filepaths_and_labels(self):
        filepaths = []
        labels = []
        class_names = sorted(os.listdir(self.directory))
        class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(self.directory, class_name)
            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if fname.endswith('.npy'):
                        filepaths.append(os.path.join(root, fname))
                        labels.append(class_indices[class_name])

        return filepaths, labels

    def __len__(self):
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        batch_filepaths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self._generate_data(batch_filepaths, batch_labels)
        return X, y

    def _generate_data(self, batch_filepaths, batch_labels):
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, filepath in enumerate(batch_filepaths):
            img = np.load(filepath)
            img = np.expand_dims(img, axis=0)
            if self.augment:
                img = self.augmentation.flow(img, batch_size=1)[0]
            X[i,] = img
            y[i] = batch_labels[i]

        return X, keras.utils.to_categorical(y, num_classes=len(set(self.labels)))

    def on_epoch_end(self):
        if self.shuffle:
            c = list(zip(self.filepaths, self.labels))
            random.shuffle(c)
            self.filepaths, self.labels = zip(*c)

    def show_augmented_example(self):
        batch_filepaths = self.filepaths[:self.batch_size]
        batch_labels = self.labels[:self.batch_size]
        X, y = self._generate_data(batch_filepaths, batch_labels)

        img = np.load(batch_filepaths[0])
        augmented_img = self.augmentation.flow(np.expand_dims(img, axis=0), batch_size=1)[0]

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Augmented Image")
        plt.imshow(augmented_img[0])
        plt.axis('off')

        plt.show()
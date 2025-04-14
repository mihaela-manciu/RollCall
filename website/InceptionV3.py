
#### KEEP ####


import os
import keras
from keras.applications import InceptionResNetV2, InceptionV3, Xception
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import seaborn as sns
from custom_data_generator2 import NumpyDataGenerator


start_time = time.time()
#
# def preprocess_and_save_images(source_folder, target_folder, target_size=(299, 299)):
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#
#     for subdir, dirs, files in os.walk(source_folder):
#         for file in files:
#             if file.endswith('.jpg'):
#                 img_path = os.path.join(subdir, file)
#                 img = cv2.imread(img_path)
#                 img = cv2.resize(img, target_size)
#                 img = img.astype('float32') / 255.0
#
#                 relative_path = os.path.relpath(subdir, source_folder)
#                 target_subdir = os.path.join(target_folder, relative_path)
#                 if not os.path.exists(target_subdir):
#                     os.makedirs(target_subdir)
#
#                 target_path = os.path.join(target_subdir, file)
#                 np.save(target_path.replace('.jpg', '.npy'), img)
#
# preprocess_and_save_images('D:/Poli/Licenta/RollCall/RollCall/Media/People', 'D:/Poli/Licenta/RollCall/RollCall/Media/Processed_People_InceptionV3')

img_rows, img_cols = 299, 299
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

for layer in base_model.layers:
    layer.trainable = False

folder_path = 'D:/Poli/Licenta/RollCall/RollCall/Media/Processed_People_InceptionV3'
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
num_classes = len(subfolders)

# custom_layer = base_model.output
# custom_layer = Flatten()(custom_layer)
# custom_layer = Dense(512, activation='relu')(custom_layer)
# custom_layer = Dropout(0.5)(custom_layer)
# custom_layer = Dense(256, activation='relu')(custom_layer)
# custom_layer = Dropout(0.3)(custom_layer)
# predictions = Dense(num_classes, activation='softmax')(custom_layer)

custom_layer = base_model.output
custom_layer = Flatten()(custom_layer)
custom_layer = Dense(256, activation='relu')(custom_layer)
predictions = Dense(num_classes, activation='softmax')(custom_layer)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

train_generator = NumpyDataGenerator(
    directory=folder_path,
    batch_size=16,
    target_size=(img_rows, img_cols),
    shuffle=True,
    augment=True,
    validation_split=0.2,
    subset='training'
)

validation_generator = NumpyDataGenerator(
    directory=folder_path,
    batch_size=8,
    target_size=(img_rows, img_cols),
    shuffle=True,
    augment=False,
    validation_split=0.2,
    subset='training'
)

# Show an example of an augmented image
# train_generator.show_augmented_example()


history = model.fit( #fit_generator
    train_generator,
    # steps_per_epoch=train_generator.samples // len(train_generator.class_indices),
    steps_per_epoch=len(train_generator), #train_generator.samples // train_generator.batch_size,
        # steps_per_epoch: represents the number of batches of samples to use for one epoch
        # calculates the number of steps based on the number of samples and the batch size
    epochs=50,
    verbose=1,
    validation_data=validation_generator,
    #validation_steps=validation_generator.samples // validation_generator.batch_size,
    # callbacks=[early_stopping, learning_rate_schedule]
    validation_steps=len(validation_generator) #validation_generator.samples // validation_generator.batch_size,
)


model.save('D:/Poli/Licenta/RollCall/RollCall/TestCNN/test_InceptionV3_0.0001_50_model.keras', overwrite=True)

end_time = time.time()

total_time_seconds = end_time - start_time

minutes = int(total_time_seconds // 60)
seconds = int(total_time_seconds % 60)

print("Model compilation time:", minutes, "minutes", seconds, "seconds")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Validation'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(['Train', 'Validation'], loc='upper left')

fig.suptitle('InceptionV3 Plots', fontsize=16)

plt.tight_layout()
plt.show()


################################
# Extract the accuracy values for each epoch - training
training_accuracy_values = history.history['accuracy']

training_average_accuracy = np.mean(training_accuracy_values)

print(f'Average Accuracy - training : {training_average_accuracy}')

##################################
# Extract the accuracy values for each epoch - validation
validation_accuracy_values = history.history['val_accuracy']

validation_average_accuracy = np.mean(validation_accuracy_values)

print(f'Average Accuracy - validation : {validation_average_accuracy}')


############################ Plot confusion matrix
validation_generator = NumpyDataGenerator(
    directory=folder_path,
    batch_size=8,
    target_size=(img_rows, img_cols),
    shuffle=False,
    augment=False
)

true_labels = []
for i in range(len(validation_generator)):
    _, labels = validation_generator[i]
    true_labels.extend(np.argmax(labels, axis=1))

predicted_labels = model.predict(validation_generator)
predicted_labels = np.argmax(predicted_labels, axis=1)

conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()















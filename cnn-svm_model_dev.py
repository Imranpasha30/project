import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from sklearn import svm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import Callback

# Suppress deprecated warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Function to load images in batches
def load_images_in_batches(folder, batch_size, label):
    filenames = os.listdir(folder)
    total_files = len(filenames)
    for idx in range(0, total_files, batch_size):
        batch_filenames = filenames[idx:min(idx + batch_size, total_files)]
        images_batch = []
        labels_batch = []
        for filename in batch_filenames:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images_batch.append(img)
                labels_batch.append(label)
        yield np.array(images_batch), np.array(labels_batch)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a custom Keras Callback for the progress bar
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs, desc='Training Progress')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix_str(f"Accuracy: {logs.get('accuracy'):.4f}, Loss: {logs.get('loss'):.4f}")

    def on_train_end(self, logs=None):
        self.progress_bar.close()

# Initialize the SVM classifier
clf = svm.SVC()

# Process the images in batches from both folders
batch_size = 1000  # Adjust the batch size according to your system's capabilities
violence_batches = load_images_in_batches('D:\\project\\outputfilr11', batch_size, 1)
non_violence_batches = load_images_in_batches('D:\\project\\outputfilr', batch_size, 0)

# Initialize lists to hold all the features and labels
features_list = []
labels_list = []

# Process violence images
for violence_batch, violence_labels in violence_batches:
    # Preprocess and reshape the batch for the CNN
    violence_batch = violence_batch / 255.0  # Example normalization
    violence_batch = violence_batch.reshape((-1, 64, 64, 3))  # Reshape for the CNN

    # Train the CNN on the batch
    progress_bar = TQDMProgressBar()
    model.fit(violence_batch, violence_labels, epochs=10, batch_size=32, callbacks=[progress_bar])

    # Extract features using the CNN
    features_train_batch = model.predict(violence_batch)

    # Flatten the features for the SVM
    features_train_batch = features_train_batch.reshape(features_train_batch.shape[0], -1)

    # Store the features and labels
    features_list.append(features_train_batch)
    labels_list.append(violence_labels)

# Process non-violence images
for non_violence_batch, non_violence_labels in non_violence_batches:
    # Preprocess and reshape the batch for the CNN
    non_violence_batch = non_violence_batch / 255.0  # Example normalization
    non_violence_batch = non_violence_batch.reshape((-1, 64, 64, 3))  # Reshape for the CNN

    # Train the CNN on the batch
    progress_bar = TQDMProgressBar()
    model.fit(non_violence_batch, non_violence_labels, epochs=10, batch_size=32, callbacks=[progress_bar])

    # Extract features using the CNN
    features_train_batch = model.predict(non_violence_batch)

    # Flatten the features for the SVM
    features_train_batch = features_train_batch.reshape(features_train_batch.shape[0], -1)

    # Store the features and labels
    features_list.append(features_train_batch)
    labels_list.append(non_violence_labels)

# Combine the stored features and labels
features = np.vstack(features_list)
labels = np.concatenate(labels_list)

# Shuffle the dataset
indices = np.arange(labels.shape[0])
np.random.shuffle(indices)
features = features[indices]
labels = labels[indices]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM classifier
clf.fit(X_train, y_train)

# Evaluate the SVM classifier
accuracy = clf.score(X_test, y_test)
print('Test accuracy:', accuracy)

# Save the trained SVM model
from joblib import dump
dump(clf, 'D:\\project\\trained_svm_model.joblib')

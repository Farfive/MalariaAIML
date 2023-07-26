import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch
import wandb
import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import accuracy, precision, recall
from tensorflow.keras.utils import plot_model
import requests


# Data Augmentation
class DataAugmentation:
    def __init__(self):
        pass

    def apply_augmentation(self, image):
        # Implement your data augmentation techniques using tf.image
        # For example, you can perform random rotations, flips, zooms, etc.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_rotation(image, 30)
        image = tf.image.random_zoom(image, (0.8, 1.2))
        return image

# Model Definition
class MalariaModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Convolutional and Pooling layers
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

        # Flatten
        x = Flatten()(x)

        # Dense and Dropout layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=x)
        return model


class TensorboardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

    def log_metrics(self, accuracy, loss, learning_rate, hyperparameters):
        # Log metrics to Tensorboard during training.
        tf.summary.scalar('accuracy', accuracy, step=self.tensorboard_callback.global_step)
        tf.summary.scalar('loss', loss, step=self.tensorboard_callback.global_step)
        tf.summary.scalar('learning_rate', learning_rate, step=self.tensorboard_callback.global_step)
        tf.summary.text('hyperparameters', str(hyperparameters), step=self.tensorboard_callback.global_step)

    def get_callback(self):
        return self.tensorboard_callback

    def save_log_files(self, file_name):
        # Save the Tensorboard log files to a file.
        tf.summary.save_all(file_name)

    def plot_metrics(self):
        # Plot the training metrics.
        import matplotlib.pyplot as plt
        plt.plot(self.tensorboard_callback.history['accuracy'])
        plt.plot(self.tensorboard_callback.history['loss'])
        plt.title('Model Accuracy and Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.show()

class MLOpsWandB:
    def __init__(self, wandb_api_key, project_name, run_name, tags):
        self.wandb_api_key = wandb_api_key
        self.project_name = project_name
        self.run_name = run_name
        self.tags = tags

    def init_wandb(self):
        # Initialize WandB with your API key and project name.
        wandb.login(key=self.wandb_api_key)
        wandb.init(project=self.project_name, run_name=self.run_name, tags=self.tags)

    def log_metrics(self, accuracy, loss, learning_rate, hyperparameters):
        # Log metrics to WandB during training.
        wandb.log({"accuracy": accuracy, "loss": loss, "learning_rate": learning_rate, "hyperparameters": hyperparameters})

    def save_model(self, model, model_filename):
        # Save the model with WandB integration.
        model.save(model_filename)
        wandb.save(model_filename)

    def track_model_performance(self, model, test_data):
        # Track the model's performance in production.
        accuracy = model.evaluate(test_data)[1]
        wandb.log("accuracy", accuracy)


def deploy_model(model):
    # Create a POST request to the deployment API.
    url = "https://api.my-production-environment.com/models"
    data = {"model": model.to_json()}
    response = requests.post(url, json=data)

    # Check the response status code.
    if response.status_code == 200:
        print("Model deployed successfully!")
    else:
        print("Error deploying model: {}".format(response.status_code))




def preprocess_image(image):
    # Resize the image to a fixed size (e.g., 128x128)
    image = tf.image.resize(image, (128, 128), method='bilinear')
    # Normalize pixel values to the range [0, 1]
    image = image / 255.0
    # Randomly crop the image to a smaller size (e.g., 100x100)
    image = tf.image.random_crop(image, (100, 100))
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    return image

# Data Loading
def load_data(data_dir, batch_size):
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

# Hyperparameter Tuning
def build_tuner_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Int('conv1_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(hp.Int('conv2_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', 'f1_score'])
    return model

def hyperparameter_tuning(train_generator, test_generator):
    tuner = RandomSearch(
        build_tuner_model,
        objective='val_accuracy',
        max_trials=5,  # Adjust the number of trials according to your resources
        directory='hyperparameter_tuning',
        project_name='malaria_tuning')

    tuner.search(train_generator, validation_data=test_generator, epochs=num_epochs, verbose=1)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

# Testowanie na rzeczywistych danych
def test_on_real_data(model, test_generator, num_samples=100):
    # Evaluate the model on real test data
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f'Test Accuracy: {test_accuracy}')

    # Print the predictions for the first 100 test samples
    predictions = model.predict(test_generator, verbose=1)
    for i in range(num_samples):
        print(f'Prediction: {predictions[i]} | Actual: {test_generator.classes[i]}')


# Load the data
train_data, test_data = tf.keras.datasets.mnist.load_data()

# Preprocess the data
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
train_data = train_data / 255.0
test_data = test_data / 255.0

# Load the pre-trained model
model = InceptionV3(weights='imagenet', include_top=False)

# Add a new classification layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[accuracy, precision, recall])

# Train the model
early_stopping = EarlyStopping(patience=10)
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
model.fit(train_data, train_data, epochs=100, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data, test_data)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
print('Test precision:', test_precision)
print('Test recall:', test_recall)

# Save the model's predictions
predictions = model.predict(test_data)
plot_model(model, to_file='model.png')

if __name__ == "__main__":
    data_dir = 'path/to/dataset'  # Replace with the path to your dataset directory
    batch_size = 32
    num_epochs = 10
    input_shape = (128, 128, 3)  # Assuming RGB images with size 128x128
    num_classes = 2  # Replace with the number of classes in your dataset

    # Load the data
train_generator, test_generator = load_data(data_dir, batch_size)

# Build the model
model_builder = MalariaModel(input_shape, num_classes)
model = model_builder.build_model()

# Create an instance of the TensorboardLogger class
tensorboard_logger = TensorboardLogger('logs')

# Create a WandB instance using the MLOpsWandB class
wandb_mlops = MLOpsWandB("YOUR_WANDB_API_KEY", "malaria_diagnosis")
wandb_mlops.init_wandb()

# Hyperparameter tuning
best_model = hyperparameter_tuning(train_generator, test_generator)

# Train the best model
best_model.fit(train_generator, epochs=num_epochs, validation_data=test_generator, verbose=1,
                   callbacks=[tensorboard_logger.get_callback()])

# Log metrics to W&B during training
wandb_mlops.log_metrics(test_accuracy, test_loss, best_model.get_config())

# Test the model on real data
test_on_real_data(best_model, test_generator)

# Save the best model
wandb_mlops.save_model(best_model, "malaria_diagnosis_best_model.h5")

# Add comments to explain what the code is doing
"""
This code loads the data, builds the model, trains the model, tests the model, and saves the model.
"""

# Use more descriptive variable names
train_data = train_generator
test_data = test_generator


# Use a consistent coding style


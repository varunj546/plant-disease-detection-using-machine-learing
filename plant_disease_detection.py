import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "dataset/PlantVillage"

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Get class names
class_names = dataset.class_names

print("Classes:")
print(class_names)

print("\nNumber of classes:", len(class_names))

# Split dataset
dataset_size = tf.data.experimental.cardinality(dataset).numpy()

train_size = int(0.8 * dataset_size)
validation_size = int(0.2 * dataset_size)

train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

# CNN Model
model = models.Sequential([

    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model
model.summary()

# Train model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Evaluate model
loss, accuracy = model.evaluate(validation_dataset)

print("\nValidation Accuracy:", accuracy)

# Save model
model.save("plant_disease_model.keras")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")

plt.show()

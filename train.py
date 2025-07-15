import tensorflow as tf;
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

# Preprocess
train_images = train_images.astype("float32") / 255.0
valid_images = valid_images.astype("float32") / 255.0
train_images = np.expand_dims(train_images, -1)  # shape: (28, 28, 1)
valid_images = np.expand_dims(valid_images, -1)

number_of_classes = train_labels.max() + 1
number_of_classes

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    train_labels,
    epochs=15,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)

val_loss, val_acc = model.evaluate(valid_images, valid_labels, verbose=1)
print(f"Validation accuracy: {val_acc:.2%}, Loss: {val_loss:.4f}")
model.save("fashion_model.keras")
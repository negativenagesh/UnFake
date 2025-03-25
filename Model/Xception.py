import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from PIL import Image

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("No GPU detected. Running on CPU.")

# Define paths to your dataset folders
real_images_path = '/home/vu-lab03-pc24/Downloads/Real'
fake_images_path = '/home/vu-lab03-pc24/Downloads/fake'
transformed_real_path = '/home/vu-lab03-pc24/Downloads/Real_transformed'
transformed_fake_path = '/home/vu-lab03-pc24/Downloads/Fake_transformed'

# Create directories for transformed images if they don’t exist
os.makedirs(transformed_real_path, exist_ok=True)
os.makedirs(transformed_fake_path, exist_ok=True)

def transform_images(input_dir, output_dir, target_size=(299, 299)):
    """
    Transform images from input directory, save to output directory,
    and return paths to transformed images.
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    print(f"Processing directory: {input_dir}")
    
    transformed_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) 
             and f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(files)} valid image files in {input_dir}")
    
    for filename in files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Load, resize, and save the image
        try:
            img = Image.open(img_path)
            img_resized = img.resize(target_size)
            img_resized.save(output_path)
            transformed_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Successfully transformed {len(transformed_paths)} images from {input_dir} to {output_dir}")
    return transformed_paths

# Transform images
real_images_transformed = transform_images(real_images_path, transformed_real_path)
fake_images_transformed = transform_images(fake_images_path, transformed_fake_path)

real_images = real_images_transformed
fake_images = fake_images_transformed

# Collect image paths and labels (0 for real, 1 for fake)
all_images = real_images + fake_images
all_labels = [0] * len(real_images) + [1] * len(fake_images)
print(f"Total images: {len(all_images)} (Real: {len(real_images)}, Fake: {len(fake_images)})")

# Split into training and validation sets (80% train, 20% validation)
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Create dataframes for train and validation
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_images, 'class': val_labels})
train_df['class'] = train_df['class'].astype(str)
val_df['class'] = val_df['class'].astype(str)

# Define image size and batch size
img_height, img_width = 299, 299  # Xception input size
batch_size = 32

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Preprocessing for validation (no augmentation)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Build the Xception-based model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification
model = Model(inputs=base_model.input, outputs=x)

# Freeze the base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model_initial.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

# Initial training
print("Starting initial training with frozen base layers...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning: Unfreeze the last 10 layers
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Update callbacks for fine-tuning
callbacks[-1] = ModelCheckpoint('best_model_finetuned.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Fine-tuning training
print("Starting fine-tuning with unfrozen layers...")
history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('deepfake_detector_final.h5')

# Prediction function
def predict_image(img_path, model):
    """Predict whether an image is Real or Fake."""
    try:
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = model.predict(img_array, verbose=0)
        return 'Fake' if prediction[0] > 0.5 else 'Real'
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Evaluate the model on validation set
val_predictions = model.predict(validation_generator)
val_pred_labels = (val_predictions > 0.5).astype(int)
val_true_labels = validation_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(val_true_labels, val_pred_labels, target_names=['Real', 'Fake']))

print("\nConfusion Matrix:")
print(confusion_matrix(val_true_labels, val_pred_labels))

# Plot training history
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history, 'Initial Training')
plot_history(history_fine, 'Fine-Tuning')

# Example usage of prediction
# test_image_path = '/home/vu-lab03-pc24/Downloads/Real_transformed/sample_image.jpg'
# result = predict_image(test_image_path, model)
# print(f"Prediction for {test_image_path}: {result}")
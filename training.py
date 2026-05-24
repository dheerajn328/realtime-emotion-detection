import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
%matplotlib inline

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

#################################################################
#                   CONFIGURATION PARAMETERS                     #
#################################################################

CONFIG = {
    'epochs': 100,
    'batch_size': 48,
    'test_split': 0.2,
    'target_size': (96, 96),
    'learning_rate': 0.0001,
    'classes': 7,
    'seed': 23,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5
}

print("="*70)
print("CK+ FACIAL EMOTION RECOGNITION - DENSENET121")
print("="*70)
print(f"Configuration:")
for key, value in CONFIG.items():
    print(f"  {key:.<30} {value}")
print(f"  Seed set to: {CONFIG['seed']}")
print("="*70)

# Set seeds for reproducibility
np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

#################################################################
#                      DATA PREPARATION                          #
#################################################################

print("\n[1/5] PREPARING DATASET")
print("-" * 70)

# Copy dataset from Drive to Colab local storage (faster access)
!cp -r "/content/drive/MyDrive/FacialEmotion/CK+" "/content/CK+"
data_dir = "/content/CK+"
model_save_path = "/content/drive/MyDrive/FacialEmotion/ck_plus_densenet_best.h5"

print(f"✓ Dataset copied to: {data_dir}")
print(f"✓ Model save path: {model_save_path}")

# Data augmentation with improved parameters
train_datagen = ImageDataGenerator(
    brightness_range=[0.8, 1.2],
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest',
    validation_split=CONFIG['test_split'],
    preprocessing_function=preprocess_input
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=CONFIG['target_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=True,
    seed=CONFIG['seed'],
    subset="training"
)

# Validation generator (no augmentation except preprocessing)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=CONFIG['test_split']
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=CONFIG['target_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False,
    seed=CONFIG['seed'],
    subset="validation"
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {val_generator.samples}")
print(f"✓ Class labels: {list(train_generator.class_indices.keys())}")

#################################################################
#                    CLASS WEIGHT CALCULATION                    #
#################################################################

print("\n[2/5] CALCULATING CLASS WEIGHTS")
print("-" * 70)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

print("✓ Class weights computed (addressing imbalance):")
for idx, (class_name, class_idx) in enumerate(train_generator.class_indices.items()):
    count = np.sum(train_generator.classes == class_idx)
    print(f"  {class_name:.<20} Weight: {class_weights[idx]:.3f} | Samples: {count}")

#################################################################
#                       MODEL BUILDING                           #
#################################################################

print("\n[3/5] BUILDING MODEL")
print("-" * 70)

# Load pre-trained DenseNet121
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(*CONFIG['target_size'], 3)
)

# Build custom classification head
x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = BatchNormalization(name='bn_1')(x)
x = Dropout(0.5, name='dropout_1')(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.001), name='fc_1')(x)
x = BatchNormalization(name='bn_2')(x)
x = Dropout(0.5, name='dropout_2')(x)
predictions = Dense(CONFIG['classes'], activation='softmax', name='predictions')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions, name='CK_Plus_DenseNet121')

# Unfreeze all layers for fine-tuning
for layer in model.layers:
    layer.trainable = True

# Compile model
optimizer = Adam(learning_rate=CONFIG['learning_rate'])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Model built successfully")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Trainable parameters: {sum([K.count_params(w) for w in model.trainable_weights]):,}")

#################################################################
#                       CALLBACKS SETUP                          #
#################################################################

callbacks = [
    ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=CONFIG['reduce_lr_patience'],
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
]

#################################################################
#                       MODEL TRAINING                           #
#################################################################

print("\n[4/5] TRAINING MODEL")
print("-" * 70)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=CONFIG['epochs'],
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed!")

#################################################################
#                    TRAINING VISUALIZATION                      #
#################################################################

print("\n[5/5] GENERATING VISUALIZATIONS")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#################################################################
#                       MODEL EVALUATION                         #
#################################################################

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Load best model
print(f"\nLoading best model from: {model_save_path}")
best_model = load_model(model_save_path)

# Evaluate
print("\nEvaluating on validation set...")
eval_results = best_model.evaluate(val_generator, verbose=0)
print(f"✓ Validation Loss: {eval_results[0]:.4f}")
print(f"✓ Validation Accuracy: {eval_results[1]:.4f} ({eval_results[1]*100:.2f}%)")

# Generate predictions
print("\nGenerating predictions...")
predictions = best_model.predict(val_generator, steps=len(val_generator), verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
target_names = list(val_generator.class_indices.keys())

# Classification Report
print("\n" + "-"*70)
print("CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names))

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    cbar_kws={'label': 'Count'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#################################################################
#                    CUSTOM IMAGE TESTING                        #
#################################################################

from google.colab import files
from tensorflow.keras.preprocessing import image

print("\n" + "="*70)
print("CUSTOM IMAGE TESTING")
print("="*70)
print("Please upload an image for emotion prediction:")

uploaded = files.upload()

if uploaded:
    image_path = list(uploaded.keys())[0]
    print(f"\n✓ Image uploaded: {image_path}")

    # Load and preprocess image
    img = image.load_img(image_path, target_size=CONFIG['target_size'])
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Predict
    prediction = best_model.predict(img_preprocessed, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    class_labels = {v: k for k, v in val_generator.class_indices.items()}
    predicted_emotion = class_labels[predicted_class_idx]
    confidence = np.max(prediction[0]) * 100

    # Display results
    print(f"\n{'='*70}")
    print(f"PREDICTION: {predicted_emotion.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"{'='*70}")

    # Show all class probabilities
    print("\nAll class probabilities:")
    for idx, prob in enumerate(prediction[0]):
        emotion = class_labels[idx]
        print(f"  {emotion:.<20} {prob*100:>6.2f}%")

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_emotion} ({confidence:.1f}% confidence)",
              fontsize=14, fontweight='bold', pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\n" + "="*70)
print("✓ SCRIPT COMPLETED SUCCESSFULLY")
print("="*70)
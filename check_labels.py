from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input

# Point to your CK+ dataset
data_dir = "/content/CK+"  # Change if running locally

# Create generator exactly like training
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset="training"
)

print("\n" + "="*70)
print("YOUR MODEL'S CLASS ORDER:")
print("="*70)
for emotion, index in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
    print(f"{index}: {emotion}")
print("="*70)

print("\nCurrent EMOTION_LABELS in webcam code:")
EMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
for i, label in enumerate(EMOTION_LABELS):
    print(f"{i}: {label}")

print("\n✓ If these match perfectly, you're good!")
print("✗ If different, update EMOTION_LABELS in webcam code!")
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import sys
import time
from collections import deque

#################################################################
#                      CONFIGURATION                             #
#################################################################

CONFIG = {
    'model_path': 'ck_plus_densenet_best.h5',
    'input_size': (96, 96),
    'camera_index': 0,
    'confidence_threshold': 50,
    'scale_factor': 1.1,  # More sensitive face detection
    'min_neighbors': 4,   # More sensitive
    'fps_smoothing': 30,
    'emotion_smoothing': 3,  # Less smoothing for faster response
    'face_padding': 0.3  # Add 30% padding around detected face
}

# CK+ emotion labels - VERIFY THIS ORDER MATCHES YOUR TRAINING!
# Print your val_generator.class_indices to confirm order
EMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Emoji mapping for better visualization
EMOTION_EMOJI = {
    'anger': '😠',
    'contempt': '😏',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sadness': '😢',
    'surprise': '😲'
}

COLORS = {
    'high_confidence': (0, 255, 0),
    'medium_confidence': (0, 255, 255),
    'low_confidence': (0, 165, 255),
    'text_bg': (0, 0, 0),
    'info_panel': (40, 40, 40)
}

#################################################################
#                    HISTOGRAM EQUALIZATION                      #
#################################################################

def enhance_face(face_roi):
    """Apply histogram equalization for better lighting consistency"""
    # Convert to LAB color space
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge and convert back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

#################################################################
#                    UTILITY FUNCTIONS                           #
#################################################################

def put_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness=2, padding=5):
    """Draw text with background"""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def get_confidence_color(confidence):
    """Return color based on confidence"""
    if confidence >= 60:
        return COLORS['high_confidence']
    elif confidence >= 40:
        return COLORS['medium_confidence']
    else:
        return COLORS['low_confidence']

#################################################################
#                    PREDICTION SMOOTHER                         #
#################################################################

class PredictionSmoother:
    """Smooth predictions with weighted average (recent frames weighted more)"""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
    
    def add_prediction(self, prediction):
        """Add new prediction with weighted smoothing"""
        self.predictions.append(prediction)
        
        if len(self.predictions) == 0:
            return prediction
        
        # Weighted average (recent predictions weighted more)
        weights = np.exp(np.linspace(-1, 0, len(self.predictions)))
        weights = weights / weights.sum()
        
        weighted_pred = np.zeros_like(prediction)
        for pred, weight in zip(self.predictions, weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def reset(self):
        self.predictions.clear()

#################################################################
#                    FPS CALCULATOR                              #
#################################################################

class FPSCalculator:
    
    def __init__(self, buffer_size=30):
        self.frame_times = deque(maxlen=buffer_size)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0.0
        return 1.0 / np.mean(self.frame_times)

#################################################################
#                    MAIN APPLICATION                            #
#################################################################

class EmotionDetector:
    
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.cap = None
        self.fps_calc = FPSCalculator(CONFIG['fps_smoothing'])
        self.smoothers = {}
        self.class_indices = None
        
    def initialize(self):
        print("=" * 70)
        print("REAL-TIME EMOTION DETECTION - CALIBRATED FOR CK+")
        print("=" * 70)
        
        # Load model
        print(f"\n[1/3] Loading model: {CONFIG['model_path']}")
        try:
            self.model = load_model(CONFIG['model_path'], compile=False)
            print("✓ Model loaded!")
            
            # CRITICAL: Print expected class order
            print("\n⚠️  VERIFY EMOTION ORDER MATCHES YOUR TRAINING:")
            print(f"   Current order: {EMOTION_LABELS}")
            print("   If wrong, update EMOTION_LABELS to match val_generator.class_indices\n")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
        
        # Load face detector
        print("[2/3] Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            print("✗ Error loading face detector")
            sys.exit(1)
        print("✓ Face detector loaded!")
        
        # Initialize camera
        print(f"[3/3] Opening camera...")
        self.cap = cv2.VideoCapture(CONFIG['camera_index'])
        
        if not self.cap.isOpened():
            print("✗ Cannot access camera")
            sys.exit(1)
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
        
        print("✓ Camera ready!")
        print("\n" + "=" * 70)
        print("CONTROLS: [Q] Quit | [S] Save | [R] Reset | [D] Debug")
        print("=" * 70)
        print("\n🎥 Starting...\n")
    
    def preprocess_face(self, face_roi):
        """CRITICAL: Match exact training preprocessing"""
        
        # Add padding around face (helps capture full expression)
        h, w = face_roi.shape[:2]
        pad = int(CONFIG['face_padding'] * min(h, w))
        face_roi = cv2.copyMakeBorder(face_roi, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        # Enhance lighting (same conditions as training)
        face_enhanced = enhance_face(face_roi)
        
        # Convert BGR -> RGB (OpenCV uses BGR, model expects RGB)
        face_rgb = cv2.cvtColor(face_enhanced, cv2.COLOR_BGR2RGB)
        
        # Resize to exact model input
        face_resized = cv2.resize(face_rgb, CONFIG['input_size'], interpolation=cv2.INTER_AREA)
        
        # Convert to array
        face_array = img_to_array(face_resized)
        face_batch = np.expand_dims(face_array, axis=0)
        
        # CRITICAL: Apply DenseNet preprocessing (ImageNet normalization)
        face_preprocessed = preprocess_input(face_batch.copy())
        
        return face_preprocessed
    
    def predict_emotion(self, face_preprocessed, face_id, debug=False):
        """Predict with smoothing"""
        
        # Raw prediction
        raw_prediction = self.model.predict(face_preprocessed, verbose=0)[0]
        
        # Initialize smoother
        if face_id not in self.smoothers:
            self.smoothers[face_id] = PredictionSmoother(CONFIG['emotion_smoothing'])
        
        # Smooth prediction
        smoothed = self.smoothers[face_id].add_prediction(raw_prediction)
        
        # Get top emotion
        emotion_idx = np.argmax(smoothed)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = smoothed[emotion_idx] * 100
        
        if debug:
            print("\n--- PREDICTION DEBUG ---")
            sorted_idx = np.argsort(smoothed)[::-1]
            for idx in sorted_idx:
                print(f"{EMOTION_LABELS[idx]:.<15} {smoothed[idx]*100:>6.2f}%")
            print("-" * 25)
        
        return emotion, confidence, smoothed
    
    def draw_info_panel(self, frame, fps):
        """Info panel"""
        height, width = frame.shape[:2]
        panel_height = 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), COLORS['info_panel'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        put_text_with_background(
            frame, "EMOTION DETECTOR", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
            COLORS['info_panel'], 2, 3
        )
        
        fps_text = f"FPS: {fps:.0f}"
        put_text_with_background(
            frame, fps_text, (width - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
            COLORS['info_panel'], 2, 3
        )
    
    def draw_emotion_bars(self, frame, x, y, h, predictions):
        """Draw compact emotion bars"""
        bar_x = x
        bar_y = y + h + 10
        bar_width = 120
        bar_height = 12
        spacing = 3
        
        # Top 4 emotions
        sorted_indices = np.argsort(predictions)[::-1][:4]
        
        for i, idx in enumerate(sorted_indices):
            emotion = EMOTION_LABELS[idx]
            prob = predictions[idx] * 100
            
            y_pos = bar_y + i * (bar_height + spacing)
            
            # Background
            cv2.rectangle(frame, 
                         (bar_x, y_pos),
                         (bar_x + bar_width, y_pos + bar_height),
                         (40, 40, 40), -1)
            
            # Filled bar
            filled = int(bar_width * predictions[idx])
            color = get_confidence_color(prob)
            cv2.rectangle(frame,
                         (bar_x, y_pos),
                         (bar_x + filled, y_pos + bar_height),
                         color, -1)
            
            # Text
            text = f"{emotion[:3].upper()} {prob:.0f}%"
            cv2.putText(frame, text,
                       (bar_x + 3, y_pos + 9),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def process_frame(self, frame, debug=False):
        """Process frame"""
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for face_id, (x, y, w, h) in enumerate(faces):
            # Extract face with margin
            margin = int(0.2 * w)
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            try:
                # Preprocess and predict
                face_prep = self.preprocess_face(face_roi)
                emotion, confidence, predictions = self.predict_emotion(face_prep, face_id, debug)
                
                # Color
                color = get_confidence_color(confidence)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Emotion label with emoji
                emoji = EMOTION_EMOJI.get(emotion, '😐')
                label = f"{emoji} {emotion.upper()}"
                
                # Large emotion text
                put_text_with_background(
                    frame, label, (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                    color, 2, 8
                )
                
                # Confidence
                conf_text = f"{confidence:.0f}%"
                put_text_with_background(
                    frame, conf_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    COLORS['text_bg'], 2, 5
                )
                
                # Emotion bars below face
                self.draw_emotion_bars(frame, x, y, h, predictions)
                
            except Exception as e:
                print(f"⚠ Error: {e}")
        
        # Face count
        put_text_with_background(
            frame, f"Faces: {len(faces)}", (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
            COLORS['text_bg'], 2, 3
        )
        
        return frame
    
    def run(self):
        """Main loop"""
        frame_count = 0
        debug_mode = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Camera read failed")
                    break
                
                # Mirror
                frame = cv2.flip(frame, 1)
                
                # FPS
                self.fps_calc.update()
                fps = self.fps_calc.get_fps()
                
                # Info panel
                self.draw_info_panel(frame, fps)
                
                # Process
                frame = self.process_frame(frame, debug_mode)
                
                # Display
                cv2.imshow('Emotion Detection - CK+ DenseNet', frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"emotion_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('r'):
                    self.smoothers.clear()
                    print("🔄 Reset")
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    print(f"🐛 Debug: {'ON' if debug_mode else 'OFF'}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⚠ Interrupted")
        finally:
            self.cleanup()
            print(f"\n✓ Processed {frame_count} frames")
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

#################################################################
#                         ENTRY POINT                            #
#################################################################

if __name__ == "__main__":
    print("\n🔧 TROUBLESHOOTING TIPS:")
    print("1. Press 'D' during runtime to see all emotion probabilities")
    print("2. Verify EMOTION_LABELS order matches your training data")
    print("3. Try smiling with EXAGGERATED expressions (CK+ has very pronounced emotions)")
    print("4. Ensure good lighting on your face")
    print("5. If still wrong, your model may need retraining with data augmentation\n")
    
    try:
        detector = EmotionDetector()
        detector.initialize()
        detector.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
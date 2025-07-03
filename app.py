from flask import Flask, render_template, Response
import cv2
import time
import queue
import threading
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pyttsx3

app = Flask(__name__)

# Load YOLOv8 Nano model (for object detection)
model = YOLO("yolov8n.pt")

# Load BLIP model (for scene captioning)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ========================== Voice Engine Setup ==========================
alert_queue = queue.Queue()
spoken_text_cache = set()  # prevent repeated speech

def speak_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # adjust speech speed
    while True:
        try:
            text = alert_queue.get()
            if text not in spoken_text_cache:
                spoken_text_cache.add(text)
                engine.say(text)
                engine.runAndWait()
                time.sleep(0.5)  # slight pause
            alert_queue.task_done()
        except Exception as e:
            print("Voice error:", e)

# Start voice engine in background thread
threading.Thread(target=speak_worker, daemon=True).start()

def speak(text):
    if text not in spoken_text_cache and not alert_queue.full():
        alert_queue.put(text)

# =========================================================================

# Cooldown variables
last_alert_time = 0
alert_cooldown = 3  # seconds
last_caption_time = 0
caption_cooldown = 10  # seconds
last_spoken_reset = time.time()

def get_caption(frame):
    """Generates a scene caption using BLIP model."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def gen_frames():
    """Video frame generator for real-time detection and alerts."""
    global last_alert_time, last_caption_time, last_spoken_reset
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        alert_triggered = False
        frame_h, frame_w = frame.shape[:2]

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                (x1, y1, x2, y2) = map(int, box.xyxy[0])

                height = y2 - y1
                width = x2 - x1
                color = (0, 255, 0)
                text = label

                # Trigger alert only if object is too close
                if height > frame_h * 0.5 or width > frame_w * 0.5:
                    alert_triggered = True
                    color = (0, 0, 255)
                    text = f"{label.upper()} TOO CLOSE!"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Speak warning if new alert
        if alert_triggered and (time.time() - last_alert_time > alert_cooldown):
            speak("Warning! Object too close!")
            last_alert_time = time.time()

        # Speak scene caption every few seconds
        if time.time() - last_caption_time > caption_cooldown:
            caption = get_caption(frame)
            cv2.putText(frame, caption, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            speak(caption)
            last_caption_time = time.time()

        # Optional: Reset spoken cache every 30 seconds
        if time.time() - last_spoken_reset > 30:
            spoken_text_cache.clear()
            last_spoken_reset = time.time()

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

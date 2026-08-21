from fastapi import FastAPI, File, UploadFile, Response
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI(title="Civic Issue Detector API")

# --- 1. LOAD MODELS ON STARTUP ---
print("Loading models...")
try:
    # This looks for the file inside the 'models' folder you just made
    civic_model = YOLO("models/civic.pt")
    print("✅ Civic Model loaded!")
except Exception as e:
    print(f"❌ Error loading Civic Model: {e}")
    civic_model = None

# If you haven't finished the water model yet, this part will just skip it safely
try:
    water_model = YOLO("models/water.pt")
    print("✅ Water Model loaded!")
    has_water_model = True
except:
    print("⚠️ Water Model not found (skipping water detection)")
    has_water_model = False

@app.get("/")
def home():
    return {"message": "Civic Eye API is running. Use POST /detect to upload an image."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not civic_model:
        return {"error": "Civic model failed to load."}

    # --- 2. READ IMAGE ---
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- 3. RUN INFERENCE ---
    # A. Run Civic Model (Potholes & Garbage)
    # Potholes (Classes 1,3,9) and Garbage (Class 4)
    civic_results = civic_model(img, conf=0.15, classes=[1, 3, 4, 9], verbose=False)

    # B. Run Water Model (if available)
    if has_water_model:
        water_results = water_model(img, conf=0.25, verbose=False)

    # --- 4. DRAW BOXES ---
    def draw_box(image, box, label, color):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw Civic Issues
    for result in civic_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 4: # Garbage
                draw_box(img, box, "Garbage", (0, 255, 255)) # Yellow
            else: # Potholes
                draw_box(img, box, "Pothole", (0, 0, 255))   # Red

    # Draw Water Issues
    if has_water_model:
        for result in water_results:
            for box in result.boxes:
                draw_box(img, box, "Water/Sewage", (255, 100, 0)) # Blue

    # --- 5. RETURN IMAGE ---
    _, encoded_img = cv2.imencode('.jpg', img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
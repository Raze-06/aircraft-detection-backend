from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import uuid
import os

app = FastAPI()

# ✅ Ensure static folder exists
os.makedirs("static", exist_ok=True)

# ✅ CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Fix model path (IMPORTANT for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)


# ==========================
# 📸 IMAGE DETECTION
# ==========================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 480))
        results = model(img)
        boxes = results[0].boxes

        aircraft = 0
        helicopters = 0
        fighter_jets = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    fighter_jets += 1
                elif cls == 1:
                    aircraft += 1
                elif cls == 2:
                    helicopters += 1

        annotated = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse({
            "aircraft": aircraft,
            "helicopters": helicopters,
            "fighter_jets": fighter_jets,
            "output_image": img_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})


# ==========================
# 🎥 VIDEO DETECTION + TRACKING
# ==========================
@app.post("/predict-video")
async def predict_video(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(await file.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)

        # Output video
        output_filename = f"output_{uuid.uuid4().hex}.mp4"
        output_path = f"static/{output_filename}"

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ✅ FIXED codec (important for deployment)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise Exception("VideoWriter failed to open")

        # Tracking sets (avoid duplicate counting)
        aircraft_ids = set()
        helicopter_ids = set()
        fighter_jet_ids = set()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ✅ Frame skipping (performance)
            if frame_count % 3 != 0:
                continue

            results = model.track(frame, persist=True, conf=0.2)
            res = results[0]

            boxes = res.boxes

            if boxes is not None and boxes.id is not None:
                for box, obj_id in zip(boxes, boxes.id):
                    cls = int(box.cls[0])
                    obj_id = int(obj_id)

                    if cls == 0:
                        fighter_jet_ids.add(obj_id)
                    elif cls == 1:
                        aircraft_ids.add(obj_id)
                    elif cls == 2:
                        helicopter_ids.add(obj_id)

            annotated = res.plot()
            out.write(annotated)

        cap.release()
        out.release()

        # ✅ Dynamic base URL (IMPORTANT for deployment)
        base_url = str(request.base_url)

        return {
            "aircraft": len(aircraft_ids),
            "helicopters": len(helicopter_ids),
            "fighter_jets": len(fighter_jet_ids),
            "video_url": f"{base_url}static/{output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}
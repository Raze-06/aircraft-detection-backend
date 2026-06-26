from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from fastapi.staticfiles import StaticFiles
import tempfile
import uuid
from pydantic import BaseModel

from tile_pipeline import (
    get_dummy_airbase_response,
    is_supported_airbase,
    run_airbase_detection,
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
# Load model
model = YOLO("best.pt")

class AirbaseDetectionRequest(BaseModel):
    area_name: str


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # Run YOLO
        results = model(img, conf=0.4)
        boxes = results[0].boxes

        aircraft = 0
        helicopters = 0
        fighter_jets = 0

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

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
    

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    try:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(await file.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)

        # Output video path
        output_filename = f"output_{uuid.uuid4().hex}.mp4"
        output_path = f"static/{output_filename}"

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("VideoWriter failed to open")

        aircraft_ids = set()
        helicopter_ids = set()
        fighter_jet_ids = set()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
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

            #  Draw boxes + IDs
            annotated = res.plot()

            out.write(annotated)

        cap.release()
        out.release()

        return {
            "aircraft": len(aircraft_ids),
            "helicopters": len(helicopter_ids),
            "fighter_jets": len(fighter_jet_ids),
            "video_url": f"http://127.0.0.1:8000/static/{output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/detect-airbase")
async def detect_airbase(payload: AirbaseDetectionRequest):
    try:
        area_name = payload.area_name.strip() if payload.area_name else ""
        if is_supported_airbase(area_name):
            return run_airbase_detection(model=model, area_name=area_name)

        return get_dummy_airbase_response(area_name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
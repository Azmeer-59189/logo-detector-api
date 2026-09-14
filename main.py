import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Logo Detection API")

# Allow your Next.js app (any origin, since this API has no user data of its
# own -- the images it processes are transient and never stored) to call
# this from the browser if needed, and always from the server side.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)

# Only report detections we're at least this confident about. Anything
# below this is treated as "no logo detected" by the app.
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.5"))

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8MB
MAX_DIMENSION = 1280


@app.get("/")
def health():
    return {"status": "ok", "classes": model.names}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()

    # Reject very large uploads outright rather than risk an out-of-memory
    # crash trying to process them. Render's free tier has only 512MB RAM,
    # and decoding a very large image can exceed that on its own.
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Please use a photo under 8MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        # For JPEGs, draft() tells the decoder to produce a smaller image
        # DIRECTLY, without ever fully decoding it at full resolution first.
        # This is what actually keeps peak memory down -- calling
        # .convert()/.thumbnail() on an already-opened image only crops
        # down *after* the expensive full-resolution decode has already
        # happened, which was the flaw in an earlier version of this fix.
        image.draft("RGB", (MAX_DIMENSION, MAX_DIMENSION))
        image = image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    # draft() only approximates the target size (and only works for JPEG),
    # so still enforce the exact cap afterward for any format.
    if max(image.size) > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    results = model(image, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        detections.append({"brand": model.names[cls_id], "confidence": confidence})

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    best = detections[0] if detections else None

    return {
        "detected": best is not None and best["confidence"] >= MIN_CONFIDENCE,
        "best_match": best,
        "all_detections": detections,
    }
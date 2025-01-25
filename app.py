from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import subprocess
from pathlib import Path
import os
import cv2
from tracker import Tracker 

app = FastAPI()

UPLOAD_DIR = Path("static/uploads")
PROCESSED_DIR = Path("static/processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

tracker = Tracker("yolov8n.pt")  

@app.get("/", response_class=HTMLResponse)
async def upload_video_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
   
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

   
    try:
        frames = tracker.track(str(file_path))
        processed_file_name = file.filename  

       
        processed_file_path = PROCESSED_DIR / processed_file_name
        print(f"Converting frames to a video...")
        tracker.convert_video(frames, str(processed_file_path))

        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "processed_video_url": f"/static/processed/{processed_file_name}"}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/processed-videos/", response_class=HTMLResponse)
async def processed_videos(request: Request):
    video_files = [{"name": file.name, "url": f"/static/processed/{file.name}"} for file in PROCESSED_DIR.glob("*") if file.suffix == ".mp4"]
    return templates.TemplateResponse("processed_videos.html", {"request": request, "video_files": video_files})

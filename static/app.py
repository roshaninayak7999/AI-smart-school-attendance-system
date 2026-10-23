import csv
import datetime
import os
import re
import shutil
import base64
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & DIRECTORIES
# ----------------------------------------------------------------------------
SCHOOL_NAME = "Sunrise Public School"
MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.05
LOW_ATTENDANCE_PERCENT = 75

# --- Admin login credentials (change these before deploying!) ---
ADMIN_CREDENTIALS = {
    "admin": "admin123",
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
EMB_DIR = DATA_DIR / "embeddings"
PHOTO_DIR = DATA_DIR / "photos"
REPORT_DIR = DATA_DIR / "reports"
STUDENTS_CSV = DATA_DIR / "students.csv"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"
registration_lock = threading.Lock()

STUDENT_FIELDS = ["roll_no", "name", "class", "registered_on"]
ATT_FIELDS = ["date", "time", "roll_no", "name", "class", "status"]

MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (100_000, [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ]),
    "face_recognition_sface_2021dec.onnx": (10_000_000, [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ]),
}

# ----------------------------------------------------------------------------
# 2. STORAGE ENGINE
# ----------------------------------------------------------------------------
def init_storage():
    for folder in (DATA_DIR, MODEL_DIR, EMB_DIR, PHOTO_DIR, REPORT_DIR, STATIC_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for path, fields in ((STUDENTS_CSV, STUDENT_FIELDS), (ATTENDANCE_CSV, ATT_FIELDS)):
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(fields)

def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f)]

def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def today_str():
    return datetime.date.today().isoformat()

def load_students():
    return read_csv(STUDENTS_CSV)

def add_student(roll, name, cls):
    if any(s["roll_no"].strip().casefold() == roll.strip().casefold() for s in load_students()):
        raise ValueError(f"Roll number {roll} already exists.")
    with open(STUDENTS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=STUDENT_FIELDS).writerow(
            {"roll_no": roll, "name": name, "class": cls, "registered_on": today_str()})

def delete_student(roll):
    rows = [s for s in load_students() if s["roll_no"] != roll]
    write_csv(STUDENTS_CSV, STUDENT_FIELDS, rows)
    for path in (EMB_DIR / f"{roll}.npy", PHOTO_DIR / f"{roll}.jpg"):
        if path.exists():
            path.unlink()

def load_attendance(date=None):
    rows = read_csv(ATTENDANCE_CSV)
    return [r for r in rows if r["date"] == date] if date else rows

def mark_attendance(student, status="Present"):
    date, when = today_str(), datetime.datetime.now().strftime("%H:%M:%S")
    rows = read_csv(ATTENDANCE_CSV)
    for r in rows:
        if r["date"] == date and r["roll_no"] == student["roll_no"]:
            if r["status"] == "Absent" and status == "Present":
                r["time"], r["status"] = when, "Present"
                write_csv(ATTENDANCE_CSV, ATT_FIELDS, rows)
                return "updated", when
            return "exists", r["time"]
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=ATT_FIELDS).writerow({
            "date": date, "time": when, "roll_no": student["roll_no"],
            "name": student["name"], "class": student["class"], "status": status})
    return "new", when

def mark_absentees():
    date = today_str()
    rows = read_csv(ATTENDANCE_CSV)
    done = {r["roll_no"] for r in rows if r["date"] == date}
    added = 0
    for s in load_students():
        if s["roll_no"] not in done:
            rows.append({"date": date, "time": "", "roll_no": s["roll_no"], "name": s["name"],
                         "class": s["class"], "status": "Absent"})
            added += 1
    if added:
        write_csv(ATTENDANCE_CSV, ATT_FIELDS, rows)
    return added

def make_summary_report():
    rows = read_csv(ATTENDANCE_CSV)
    total_days = len({r["date"] for r in rows})
    present = defaultdict(int)
    for r in rows:
        if r["status"] == "Present":
            present[r["roll_no"]] += 1
    out = []
    for s in load_students():
        days = present[s["roll_no"]]
        percent = round(100 * days / total_days, 1) if total_days else 0.0
        out.append({"roll_no": s["roll_no"], "name": s["name"], "class": s["class"],
                    "days_present": days, "total_days": total_days, "percentage": percent,
                    "remark": "Low attendance" if total_days and percent < LOW_ATTENDANCE_PERCENT else ""})
    path = REPORT_DIR / f"attendance_summary_{today_str()}.csv"
    write_csv(path, ["roll_no", "name", "class", "days_present", "total_days", "percentage", "remark"], out)
    return path

# ----------------------------------------------------------------------------
# 3. FACE AI ENGINE
# ----------------------------------------------------------------------------
def ensure_models():
    import urllib.request
    paths = []
    for filename, (min_size, urls) in MODEL_FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size >= min_size:
            paths.append(dest)
            continue
        print(f"Downloading {filename}...")
        ok = False
        for url in urls:
            try:
                tmp = dest.with_suffix(".part")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                    shutil.copyfileobj(resp, out)
                if tmp.stat().st_size < min_size:
                    raise IOError("Downloaded file is too small")
                tmp.replace(dest)
                ok = True
                break
            except Exception as e:
                print(f"Fetch failed: {e}")
        if not ok:
            raise RuntimeError(f"Could not download {filename}")
        paths.append(dest)
    return paths

class FaceEngine:
    def __init__(self):
        yunet, sface = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.85, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)

    def embed(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        vec = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

class FaceDB:
    def __init__(self):
        self.reload()

    def reload(self):
        students = {s["roll_no"]: s for s in load_students()}
        mats, owners = [], []
        for roll in students:
            path = EMB_DIR / f"{roll}.npy"
            if path.exists():
                emb = np.load(path)
                mats.append(emb)
                owners += [roll] * len(emb)
        matrix = np.vstack(mats).astype(np.float32) if mats else np.empty((0, 128), np.float32)
        self.students = students
        self.matrix = matrix
        self.owners = owners

    def identify(self, emb):
        if not self.owners:
            return None, 0.0
        scores = self.matrix @ emb
        best = {}
        for owner, sc in zip(self.owners, scores):
            if sc > best.get(owner, -1.0):
                best[owner] = float(sc)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        roll, score = ranked[0]
        if score < MATCH_THRESHOLD:
            return None, score
        if len(ranked) > 1 and score - ranked[1][1] < MATCH_MARGIN:
            return None, score
        return roll, score

    def find_duplicate(self, emb):
        if not self.owners:
            return None
        scores = self.matrix @ emb
        best = {}
        for owner, score in zip(self.owners, scores):
            if score > best.get(owner, -1.0):
                best[owner] = float(score)
        if not best:
            return None
        roll, score = max(best.items(), key=lambda item: item[1])
        return roll if score >= MATCH_THRESHOLD else None

init_storage()
engine = FaceEngine()
face_db = FaceDB()

# ----------------------------------------------------------------------------
# 4. FASTAPI APP & ENDPOINTS
# ----------------------------------------------------------------------------
app = FastAPI(title="Smart Attendance System API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.mount("/photos", StaticFiles(directory=str(PHOTO_DIR)), name="photos")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def decode_image_base64(b64_str: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    raw = base64.b64decode(b64_str)
    nparr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

class FramePayload(BaseModel):
    image: str

class LoginPayload(BaseModel):
    username: str
    password: str

@app.get("/", response_class=HTMLResponse)
def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Frontend index.html not found in static/ directory.</h1>"

@app.post("/api/login")
def login(payload: LoginPayload):
    username = (payload.username or "").strip()
    password = payload.password or ""
    expected = ADMIN_CREDENTIALS.get(username)
    if expected is None or expected != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"success": True, "username": username}

@app.get("/api/dashboard")
def get_dashboard():
    students = load_students()
    today_records = load_attendance(today_str())
    present_set = {r["roll_no"] for r in today_records if r["status"] == "Present"}
    absent_set = {r["roll_no"] for r in today_records if r["status"] == "Absent"}

    return {
        "school_name": SCHOOL_NAME,
        "total_students": len(students),
        "present_today": len(present_set),
        "absent_today": len(absent_set),
        "unmarked_today": max(0, len(students) - len(present_set) - len(absent_set)),
        "recent_attendance": sorted(
            [r for r in today_records if r["status"] == "Present"],
            key=lambda x: x["time"],
            reverse=True
        )[:15]
    }

@app.post("/api/recognize-frame")
def recognize_frame(payload: FramePayload):
    frame = decode_image_base64(payload.image)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame format")

    faces = engine.detect(frame)
    detected_list = []
    matched_student = None

    for face in faces[:5]:
        box = [int(v) for v in face[:4]]
        emb = engine.embed(frame, face)
        roll, score = face_db.identify(emb)

        if roll and roll in face_db.students:
            student = face_db.students[roll]
            detected_list.append({
                "box": box,
                "label": f"{student['name']} ({score:.2f})",
                "matched": True,
                "roll": roll
            })
            if matched_student is None:
                matched_student = student
        else:
            detected_list.append({
                "box": box,
                "label": "Unknown",
                "matched": False,
                "roll": None
            })

    mark_info = None
    if matched_student:
        status, when = mark_attendance(matched_student, "Present")
        has_photo = (PHOTO_DIR / f"{matched_student['roll_no']}.jpg").exists()
        mark_info = {
            "status": status,
            "when": when,
            "student": matched_student,
            "photo_url": f"/photos/{matched_student['roll_no']}.jpg" if has_photo else None
        }

    return {"faces": detected_list, "match": mark_info}

@app.post("/api/register-samples")
def register_samples(roll: str = Form(...), name: str = Form(...), cls: str = Form(...), frames: list[str] = Form(...)):
    roll = roll.strip()
    name = name.strip()
    cls = cls.strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", roll):
        raise HTTPException(status_code=400, detail="Roll number must contain letters, numbers, '-' or '_' only")
    if not name or not cls:
        raise HTTPException(status_code=400, detail="Student name and class are required")
    if any(s["roll_no"].strip().casefold() == roll.casefold() for s in load_students()):
        raise HTTPException(status_code=400, detail=f"Roll number {roll} already exists")

    embeddings = []
    photo = None

    for b64 in frames:
        frame = decode_image_base64(b64)
        if frame is None:
            continue
        faces = engine.detect(frame)
        if len(faces) == 1:
            face = faces[0]
            emb = engine.embed(frame, face)
            embeddings.append(emb)
            if photo is None:
                x, y, w, h = [int(v) for v in face[:4]]
                mx, my = int(w * 0.35), int(h * 0.35)
                x0, y0 = max(0, x - mx), max(0, y - my)
                crop = frame[y0:y + h + my, x0:x + w + mx]
                if crop.size > 0:
                    scale = 200 / max(crop.shape[:2])
                    photo = cv2.resize(crop, None, fx=scale, fy=scale)

    if len(embeddings) < 3:
        raise HTTPException(status_code=400, detail="Could not capture enough clear face angles. Try again.")

    emb_arr = np.array(embeddings, dtype=np.float32)
    mean_emb = emb_arr.mean(axis=0)
    mean_emb /= np.linalg.norm(mean_emb)

    with registration_lock:
        if any(s["roll_no"].strip().casefold() == roll.casefold() for s in load_students()):
            raise HTTPException(status_code=400, detail=f"Roll number {roll} already exists")
        dup_roll = face_db.find_duplicate(mean_emb)
        if dup_roll:
            other = face_db.students[dup_roll]
            raise HTTPException(status_code=400, detail=f"This face is already registered to {other['name']} (Roll: {dup_roll})")

        embedding_path = EMB_DIR / f"{roll}.npy"
        photo_path = PHOTO_DIR / f"{roll}.jpg"
        previous_students = load_students()
        student_write_started = False
        try:
            np.save(embedding_path, emb_arr)
            if photo is not None and not cv2.imwrite(str(photo_path), photo):
                raise OSError("Could not save the student photo")
            student_write_started = True
            add_student(roll, name, cls)
            face_db.reload()
        except Exception:
            if student_write_started:
                write_csv(STUDENTS_CSV, STUDENT_FIELDS, previous_students)
            for path in (embedding_path, photo_path):
                if path.exists():
                    path.unlink()
            raise
    return {"message": f"Successfully registered {name}"}

@app.get("/api/students/check-roll")
def check_roll_number(roll: str):
    normalized_roll = roll.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized_roll):
        return {"available": False, "message": "Use letters, numbers, '-' or '_' only."}
    exists = any(
        student["roll_no"].strip().casefold() == normalized_roll.casefold()
        for student in load_students()
    )
    return {
        "available": not exists,
        "message": f"Roll number {normalized_roll} is already registered." if exists else "Roll number is available."
    }

@app.get("/api/records")
def get_records(date: Optional[str] = None):
    target = date if date else today_str()
    records = load_attendance(target)
    return {"date": target, "records": sorted(records, key=lambda x: (x["status"] != "Present", x["name"]))}

@app.get("/api/students")
def get_students():
    students = load_students()
    for s in students:
        s["has_face"] = (EMB_DIR / f"{s['roll_no']}.npy").exists()
        s["photo_url"] = f"/photos/{s['roll_no']}.jpg" if (PHOTO_DIR / f"{s['roll_no']}.jpg").exists() else None
    return students

@app.delete("/api/students/{roll}")
def delete_student_api(roll: str):
    delete_student(roll)
    face_db.reload()
    return {"message": "Student deleted"}

@app.post("/api/mark-absentees")
def mark_absentees_api():
    count = mark_absentees()
    return {"count": count, "message": f"{count} students marked absent."}

@app.get("/api/export-report")
def export_report_api():
    path = make_summary_report()
    return FileResponse(path, filename=path.name, media_type="text/csv")

# ----------------------------------------------------------------------------
# 5. DIRECT EXECUTION SUPPORT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\nStarting Smart Attendance Server at http://localhost:8000 ...\n")
    print(f"Admin login -> username: '{list(ADMIN_CREDENTIALS.keys())[0]}'  (change ADMIN_CREDENTIALS in app.py)\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

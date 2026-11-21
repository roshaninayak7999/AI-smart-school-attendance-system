import base64
import csv
import datetime
import io
import os
import re
import shutil
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
SCHOOL_NAME = os.getenv("SCHOOL_NAME", "Sunrise Public School")
MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.05
LOW_ATTENDANCE_PERCENT = 75

ADMIN_CREDENTIALS = {
    os.getenv("ADMIN_USERNAME", "admin"): os.getenv("ADMIN_PASSWORD", "admin123"),
}

DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

for folder in (MODEL_DIR, STATIC_DIR):
    folder.mkdir(parents=True, exist_ok=True)

registration_lock = threading.Lock()

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
# 2. POSTGRESQL DATABASE SETUP
# ----------------------------------------------------------------------------
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable set nahi hai!")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                roll_no TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class TEXT NOT NULL,
                registered_on TEXT NOT NULL,
                embedding BYTEA NOT NULL,
                photo BYTEA
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                date TEXT NOT NULL,
                time TEXT NOT NULL DEFAULT '',
                roll_no TEXT NOT NULL REFERENCES students(roll_no) ON DELETE CASCADE,
                name TEXT NOT NULL,
                class TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE (date, roll_no)
            );
        """)
        conn.commit()


def get_ist_now():
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    return datetime.datetime.now(ist)


def today_str():
    return get_ist_now().date().isoformat()


def to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def from_bytes(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(bytes(raw)))


# ----------------------------------------------------------------------------
# 3. DATABASE QUERIES
# ----------------------------------------------------------------------------
def load_students():
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT roll_no, name, class, registered_on,
                   (embedding IS NOT NULL) AS has_face,
                   (photo IS NOT NULL) AS has_photo
            FROM students
            ORDER BY roll_no;
        """)
        rows = cur.fetchall()
    return [
        {
            "roll_no": r["roll_no"],
            "name": r["name"],
            "class": r["class"],
            "registered_on": r["registered_on"],
            "has_face": r["has_face"],
            "photo_url": f"/photos/{r['roll_no']}.jpg" if r["has_photo"] else None,
        }
        for r in rows
    ]


def roll_exists(roll: str) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM students WHERE LOWER(roll_no) = LOWER(%s);", (roll.strip(),))
        return cur.fetchone() is not None


def load_attendance(date: Optional[str] = None):
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if date:
            cur.execute(
                "SELECT date, time, roll_no, name, class, status FROM attendance WHERE date = %s;",
                (date,),
            )
        else:
            cur.execute("SELECT date, time, roll_no, name, class, status FROM attendance;")
        return [dict(r) for r in cur.fetchall()]


def mark_attendance(student: dict, status: str = "Present"):
    date = today_str()
    when = get_ist_now().strftime("%H:%M:%S")
    roll = student["roll_no"]

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, status, time FROM attendance WHERE date = %s AND roll_no = %s;",
            (date, roll),
        )
        row = cur.fetchone()
        if row:
            rec_id, old_status, old_time = row
            if old_status == "Absent" and status == "Present":
                cur.execute(
                    "UPDATE attendance SET time = %s, status = 'Present' WHERE id = %s;",
                    (when, rec_id),
                )
                conn.commit()
                return "updated", when
            return "exists", old_time or when

        cur.execute(
            """
            INSERT INTO attendance (date, time, roll_no, name, class, status)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (date, when, roll, student["name"], student["class"], status),
        )
        conn.commit()
        return "new", when


def mark_absentees():
    date = today_str()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attendance (date, time, roll_no, name, class, status)
            SELECT %s, '', s.roll_no, s.name, s.class, 'Absent'
            FROM students s
            WHERE NOT EXISTS (
                SELECT 1 FROM attendance a
                WHERE a.date = %s AND a.roll_no = s.roll_no
            );
            """,
            (date, date),
        )
        added = cur.rowcount
        conn.commit()
    return added


def make_summary_csv() -> str:
    rows = load_attendance()
    total_days = len({r["date"] for r in rows if r["date"]})
    present = defaultdict(int)
    for r in rows:
        if r["status"] == "Present":
            present[str(r["roll_no"]).strip()] += 1

    output = io.StringIO()
    fields = ["roll_no", "name", "class", "days_present", "total_days", "percentage", "remark"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    for s in load_students():
        days = present[str(s["roll_no"]).strip()]
        percent = round(100 * days / total_days, 1) if total_days else 0.0
        writer.writerow({
            "roll_no": s["roll_no"],
            "name": s["name"],
            "class": s["class"],
            "days_present": days,
            "total_days": total_days,
            "percentage": percent,
            "remark": "Low attendance" if total_days and percent < LOW_ATTENDANCE_PERCENT else "",
        })
    return output.getvalue()


# ----------------------------------------------------------------------------
# 4. FACE AI ENGINE
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
        students = {}
        mats, owners = [], []
        with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT roll_no, name, class, registered_on, embedding, (photo IS NOT NULL) AS has_photo FROM students;")
            for r in cur.fetchall():
                roll = r["roll_no"]
                students[roll] = {
                    "roll_no": roll,
                    "name": r["name"],
                    "class": r["class"],
                    "registered_on": r["registered_on"],
                    "photo_url": f"/photos/{roll}.jpg" if r["has_photo"] else None,
                }
                if r["embedding"]:
                    emb = from_bytes(r["embedding"])
                    mats.append(emb)
                    owners += [roll] * len(emb)

        self.students = students
        self.matrix = np.vstack(mats).astype(np.float32) if mats else np.empty((0, 128), np.float32)
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


init_db()
engine = FaceEngine()
face_db = FaceDB()

# ----------------------------------------------------------------------------
# 5. FASTAPI APP & ENDPOINTS
# ----------------------------------------------------------------------------
app = FastAPI(title="Smart Attendance System API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
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


@app.get("/photos/{roll}.jpg")
def get_student_photo(roll: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT photo FROM students WHERE roll_no = %s;", (roll,))
        row = cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Photo not found")
    return Response(content=bytes(row[0]), media_type="image/jpeg")


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
            reverse=True,
        )[:15],
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
                "roll": roll,
            })
            if matched_student is None:
                matched_student = student
        else:
            detected_list.append({
                "box": box,
                "label": "Unknown",
                "matched": False,
                "roll": None,
            })

    mark_info = None
    if matched_student:
        status, when = mark_attendance(matched_student, "Present")
        mark_info = {
            "status": status,
            "when": when,
            "student": matched_student,
            "photo_url": matched_student.get("photo_url"),
        }

    return {"faces": detected_list, "match": mark_info}


@app.post("/api/register-samples")
def register_samples(
    roll: str = Form(...),
    name: str = Form(...),
    cls: str = Form(...),
    frames: list[str] = Form(...),
):
    roll = roll.strip()
    name = name.strip()
    cls = cls.strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", roll):
        raise HTTPException(status_code=400, detail="Roll number must contain letters, numbers, '-' or '_' only")
    if not name or not cls:
        raise HTTPException(status_code=400, detail="Student name and class are required")
    if roll_exists(roll):
        raise HTTPException(status_code=400, detail=f"Roll number {roll} already exists")

    embeddings = []
    photo_bytes = None

    for b64 in frames:
        frame = decode_image_base64(b64)
        if frame is None:
            continue
        faces = engine.detect(frame)
        if len(faces) == 1:
            face = faces[0]
            emb = engine.embed(frame, face)
            embeddings.append(emb)
            if photo_bytes is None:
                x, y, w, h = [int(v) for v in face[:4]]
                mx, my = int(w * 0.35), int(h * 0.35)
                x0, y0 = max(0, x - mx), max(0, y - my)
                crop = frame[y0:y + h + my, x0:x + w + mx]
                if crop.size > 0:
                    scale = 200 / max(crop.shape[:2])
                    resized = cv2.resize(crop, None, fx=scale, fy=scale)
                    ok, encoded = cv2.imencode(".jpg", resized)
                    if ok:
                        photo_bytes = encoded.tobytes()

    if len(embeddings) < 3:
        raise HTTPException(status_code=400, detail="Could not capture enough clear face angles. Try again.")

    emb_arr = np.array(embeddings, dtype=np.float32)
    mean_emb = emb_arr.mean(axis=0)
    mean_emb /= np.linalg.norm(mean_emb)

    with registration_lock:
        if roll_exists(roll):
            raise HTTPException(status_code=400, detail=f"Roll number {roll} already exists")
        dup_roll = face_db.find_duplicate(mean_emb)
        if dup_roll:
            other = face_db.students[dup_roll]
            raise HTTPException(
                status_code=400,
                detail=f"This face is already registered to {other['name']} (Roll: {dup_roll})",
            )

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO students (roll_no, name, class, registered_on, embedding, photo)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (
                    roll,
                    name,
                    cls,
                    today_str(),
                    psycopg2.Binary(to_bytes(emb_arr)),
                    psycopg2.Binary(photo_bytes) if photo_bytes else None,
                ),
            )
            conn.commit()
        face_db.reload()

    return {"message": f"Successfully registered {name}"}


@app.get("/api/students/check-roll")
def check_roll_number(roll: str):
    normalized_roll = roll.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized_roll):
        return {"available": False, "message": "Use letters, numbers, '-' or '_' only."}
    exists = roll_exists(normalized_roll)
    return {
        "available": not exists,
        "message": f"Roll number {normalized_roll} is already registered." if exists else "Roll number is available.",
    }


@app.get("/api/records")
def get_records(date: Optional[str] = None):
    target = date if date else today_str()
    records = load_attendance(target)
    return {"date": target, "records": sorted(records, key=lambda x: (x["status"] != "Present", x["name"]))}


@app.get("/api/students")
def get_students():
    return load_students()


@app.delete("/api/students/{roll}")
def delete_student_api(roll: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM students WHERE roll_no = %s;", (roll,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        conn.commit()
    face_db.reload()
    return {"message": "Student deleted"}


@app.post("/api/mark-absentees")
def mark_absentees_api():
    count = mark_absentees()
    return {"count": count, "message": f"{count} students marked absent."}


@app.get("/api/export-report")
def export_report_api():
    csv_data = make_summary_csv()
    filename = f"attendance_summary_{today_str()}.csv"
    return Response(
        content=csv_data.encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

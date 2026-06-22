"""
Photo upload & processing router
"""
import secrets
import string
import uuid
import time
import threading
from datetime import datetime, timezone
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response
from database import get_supabase
from schemas import ProcessPhotoRequest
from config import settings

router = APIRouter(prefix="/photobooth", tags=["photos"])

CLEANUP_DELAY_SECONDS = 300  # 5 minutes


def _generate_download_code(length: int = 6) -> str:
    return "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length))


@router.post("/upload")
async def upload_photo(
    session_id: str = Form(...),
    photo: UploadFile = File(...),
):
    """Upload a photo to a session, then detect face expression before responding."""
    db = get_supabase()

    # Verify session exists and is active
    session_result = db.table("photo_sessions").select("id, status, expires_at").eq("id", session_id).execute()
    if not session_result.data:
        raise HTTPException(404, "Session not found")
    session = session_result.data[0]
    if session["status"] != "active":
        raise HTTPException(400, "Session is not active")
    if session.get("expires_at"):
        expires_at = datetime.fromisoformat(session["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires_at:
            db.table("photo_sessions").update({"status": "expired"}).eq("id", session_id).execute()
            raise HTTPException(400, "Session has expired")

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if photo.content_type and photo.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type. Allowed: {', '.join(allowed_types)}")

    # Validate file size
    contents = await photo.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(400, f"File too large. Max {settings.max_upload_size_mb}MB")

    # Count existing photos in session
    count_result = db.table("photos").select("id", count="exact").eq("session_id", session_id).execute()
    photo_number = (count_result.count or 0) + 1

    # Upload to Supabase Storage
    file_ext = photo.filename.split(".")[-1] if photo.filename and "." in photo.filename else "png"
    storage_path = f"{session_id}/{uuid.uuid4().hex}.{file_ext}"

    db.storage.from_("photos").upload(
        storage_path,
        contents,
        {"content-type": photo.content_type or "image/png"},
    )

    # Get public URL
    original_url = db.storage.from_("photos").get_public_url(storage_path)

    # Insert photo record
    photo_row = {
        "session_id": session_id,
        "original_path": storage_path,
        "original_url": original_url,
        "photo_number": photo_number,
        "status": "uploaded",
    }
    result = db.table("photos").insert(photo_row).execute()
    if not result.data:
        raise HTTPException(500, "Failed to save photo record")

    photo_data = result.data[0]
    photo_id = photo_data["id"]

    # Detect face expression SYNCHRONOUSLY — must finish and be saved to DB
    # before this endpoint responds, so the frontend can rely on it being
    # present right away (no race condition with /process).
    expression_result = _detect_and_save_face_expression(photo_id, contents)

    return {
        "success": True,
        "message": "Photo uploaded",
        "data": {
            "photo_id": photo_id,
            "session_id": session_id,
            "original_url": original_url,
            "photo_number": photo_number,
            "status": "uploaded",
            "face_expression": expression_result["expression"],
            "face_expression_label": expression_result["expression_label"],
            "face_confidence": expression_result["confidence"],
        },
    }


def _detect_and_save_face_expression(photo_id: str, image_bytes: bytes) -> dict:
    """
    Run face expression detection and persist the result to the database.

    Called synchronously from /upload so the expression is guaranteed to be
    saved before the upload response is returned.

    Returns a dict with the best detection (or all-None values if no face
    was detected / detection failed) so the caller can include it in the
    response without a second DB round-trip.
    """
    empty_result = {"expression": None, "expression_label": None, "confidence": None}

    try:
        from ml.face_expression import detect_expression
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        expressions = detect_expression(img)

        if not expressions:
            return empty_result

        db = get_supabase()
        best = expressions[0]
        bbox = best.get("bbox", [0, 0, 0, 0])

        # Save to face_expressions table
        db.table("face_expressions").insert({
            "photo_id": photo_id,
            "expression": best["expression"],
            "confidence": best["confidence"],
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
        }).execute()

        # Update photo with primary expression
        db.table("photos").update({
            "face_expression": best["expression"],
            "face_confidence": best["confidence"],
        }).eq("id", photo_id).execute()

        return {
            "expression": best["expression"],
            "expression_label": best.get("expression_label", best["expression"]),
            "confidence": best["confidence"],
        }

    except Exception as e:
        # Face expression is a best-effort enhancement — never block the
        # upload flow if detection or the API call fails.
        print(f"Face expression detection failed: {e}")
        return empty_result


@router.post("/process")
async def process_photo(body: ProcessPhotoRequest):
    """Start photo processing (returns immediately, runs ML in background)."""
    db = get_supabase()

    # Get photo
    photo_result = db.table("photos").select("*").eq("id", body.photo_id).execute()
    if not photo_result.data:
        raise HTTPException(404, "Photo not found")
    photo = photo_result.data[0]

    # Get background
    bg_result = db.table("backgrounds").select("*").eq("id", body.background_id).execute()
    if not bg_result.data:
        raise HTTPException(404, "Background not found")
    background = bg_result.data[0]

    # Get mascot
    mascot_result = db.table("mascots").select("*").eq("id", body.mascot_id).execute()
    if not mascot_result.data:
        raise HTTPException(404, "Mascot not found")
    mascot = mascot_result.data[0]

    # Get filter (optional)
    ai_filter = None
    if body.filter_id:
        filter_result = db.table("ai_filters").select("*").eq("id", body.filter_id).execute()
        if filter_result.data:
            ai_filter = filter_result.data[0]

    # Create processing record
    processing_row = {
        "photo_id": body.photo_id,
        "background_id": body.background_id,
        "mascot_id": body.mascot_id,
        "filter_id": body.filter_id,
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    proc_result = db.table("photo_processing").insert(processing_row).execute()
    if not proc_result.data:
        raise HTTPException(500, "Failed to create processing record")

    processing = proc_result.data[0]
    processing_id = processing["id"]

    # Run ML pipeline in background thread
    thread = threading.Thread(
        target=_run_ml_pipeline,
        args=(processing_id, photo, background, mascot, ai_filter, body.photo_id),
        daemon=True,
    )
    thread.start()

    return {
        "success": True,
        "message": "Processing started",
        "data": {
            "processing_id": processing_id,
            "photo_id": body.photo_id,
            "status": "pending",
        },
    }


def _run_ml_pipeline(
    processing_id: str,
    photo: dict,
    background: dict,
    mascot: dict,
    ai_filter: dict | None,
    photo_id: str,
):
    """Run the full ML pipeline in a background thread."""
    db = get_supabase()

    try:
        start_time = time.time()

        # Update status: bg_removal
        db.table("photo_processing").update({"status": "bg_removal"}).eq("id", processing_id).execute()

        from ml.AI_Enhancement import process_photobooth
        from PIL import Image
        from io import BytesIO
        import requests as http_requests

        # Download original photo
        photo_url = photo["original_url"]
        resp = http_requests.get(photo_url, timeout=30)
        person_img = Image.open(BytesIO(resp.content))

        # Download background
        bg_url = db.storage.from_("assets").get_public_url(background["image_path"])
        bg_resp = http_requests.get(bg_url, timeout=30)
        bg_img = Image.open(BytesIO(bg_resp.content))

        # Download mascot
        mascot_url = db.storage.from_("assets").get_public_url(mascot["image_path"])
        mascot_resp = http_requests.get(mascot_url, timeout=30)
        mascot_img = Image.open(BytesIO(mascot_resp.content))

        # Update status: compositing
        db.table("photo_processing").update({"status": "compositing"}).eq("id", processing_id).execute()

        # Process
        filter_config = ai_filter.get("config", {}) if ai_filter else {}
        result_img, ai_processed = process_photobooth(person_img, bg_img, mascot_img, filter_config)

        # Update status: ai_enhance
        db.table("photo_processing").update({"status": "ai_enhance"}).eq("id", processing_id).execute()

        # Save result to storage
        output_buffer = BytesIO()
        result_img.save(output_buffer, format="PNG", quality=98, optimize=True)
        output_bytes = output_buffer.getvalue()

        result_path = f"{photo['session_id']}/{uuid.uuid4().hex}_processed.png"
        db.storage.from_("processed").upload(
            result_path,
            output_bytes,
            {"content-type": "image/png"},
        )
        processed_url = db.storage.from_("processed").get_public_url(result_path)

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update status: generating QR
        db.table("photo_processing").update({"status": "generating_qr"}).eq("id", processing_id).execute()

        # Generate download code & QR
        download_code = _generate_download_code()
        download_url = processed_url

        from ml.qr_generator import generate_qr_code

        qr_bytes = generate_qr_code(download_url)
        qr_path = f"{photo['session_id']}/{download_code}_qr.png"
        db.storage.from_("qrcodes").upload(
            qr_path,
            qr_bytes,
            {"content-type": "image/png"},
        )
        qr_url = db.storage.from_("qrcodes").get_public_url(qr_path)

        # Update processing record → completed
        db.table("photo_processing").update({
            "status": "completed",
            "processed_path": result_path,
            "processed_url": processed_url,
            "processing_time_ms": processing_time_ms,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", processing_id).execute()

        # Update photo status
        db.table("photos").update({"status": "processed"}).eq("id", photo_id).execute()

        # Create download link
        db.table("download_links").insert({
            "processing_id": processing_id,
            "download_code": download_code,
            "qr_code_path": qr_path,
            "qr_code_url": qr_url,
            "download_url": download_url,
        }).execute()

        # Schedule auto-cleanup after 5 minutes
        session_id = photo["session_id"]
        _schedule_cleanup(session_id, processing_id, photo_id, result_path, qr_path)

        print(f"[Process] Completed {processing_id} in {processing_time_ms}ms")

    except Exception as e:
        # Mark as failed
        db.table("photo_processing").update({
            "status": "failed",
            "error_message": str(e)[:500],
        }).eq("id", processing_id).execute()
        db.table("photos").update({"status": "failed"}).eq("id", photo_id).execute()
        print(f"[Process] Failed {processing_id}: {e}")


# ── Status mapping for progress ──

STATUS_PROGRESS = {
    "pending": {"progress": 5, "text": "Memulai proses..."},
    "bg_removal": {"progress": 15, "text": "Menghapus background..."},
    "compositing": {"progress": 40, "text": "Menggabungkan dengan latar & maskot..."},
    "ai_enhance": {"progress": 65, "text": "AI Enhancement sedang berjalan..."},
    "generating_qr": {"progress": 90, "text": "Membuat QR Code..."},
    "completed": {"progress": 100, "text": "Selesai!"},
    "failed": {"progress": 0, "text": "Gagal memproses foto."},
}


@router.get("/processing/{processing_id}/status")
async def get_processing_status(processing_id: str):
    """Poll processing status (real-time progress)."""
    db = get_supabase()

    result = (
        db.table("photo_processing")
        .select("id, status, processed_url, processing_time_ms, error_message")
        .eq("id", processing_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Processing record not found")

    proc = result.data[0]
    status = proc["status"]
    info = STATUS_PROGRESS.get(status, {"progress": 0, "text": status})

    data = {
        "processing_id": proc["id"],
        "status": status,
        "progress": info["progress"],
        "status_text": info["text"],
    }

    # If completed, include result data
    if status == "completed":
        dl_result = (
            db.table("download_links")
            .select("*")
            .eq("processing_id", processing_id)
            .execute()
        )
        link = dl_result.data[0] if dl_result.data else None
        data.update({
            "processed_url": proc["processed_url"],
            "processing_time_ms": proc["processing_time_ms"],
            "download_code": link["download_code"] if link else None,
            "download_url": link["download_url"] if link else None,
            "qr_code_url": link["qr_code_url"] if link else None,
        })

    # If failed, include error
    if status == "failed":
        data["error_message"] = proc.get("error_message", "Unknown error")

    return {"success": True, "message": "Status retrieved", "data": data}


@router.get("/photo/{photo_id}")
async def get_photo(photo_id: str):
    """Get photo details with processing info."""
    db = get_supabase()

    photo_result = db.table("photos").select("*").eq("id", photo_id).execute()
    if not photo_result.data:
        raise HTTPException(404, "Photo not found")

    photo = photo_result.data[0]

    # Get processing records
    proc_result = (
        db.table("photo_processing")
        .select("*, download_links(*)")
        .eq("photo_id", photo_id)
        .execute()
    )

    # Get face expressions
    expr_result = db.table("face_expressions").select("*").eq("photo_id", photo_id).execute()

    return {
        "success": True,
        "message": "Photo retrieved",
        "data": {
            **photo,
            "processing": proc_result.data or [],
            "face_expressions": expr_result.data or [],
        },
    }


@router.get("/download/{download_code}")
async def download_by_code(download_code: str):
    """Get download URL by code."""
    db = get_supabase()

    result = (
        db.table("download_links")
        .select("*")
        .eq("download_code", download_code)
        .eq("is_active", True)
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Download link not found or expired")

    link = result.data[0]

    # Check expiry
    if link.get("expires_at"):
        expires_at = datetime.fromisoformat(link["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires_at:
            db.table("download_links").update({"is_active": False}).eq("id", link["id"]).execute()
            raise HTTPException(410, "Download link has expired")

    if link["download_count"] >= link["max_downloads"]:
        raise HTTPException(410, "Download limit reached")

    # Increment download count
    db.table("download_links").update({
        "download_count": link["download_count"] + 1,
    }).eq("id", link["id"]).execute()

    return {
        "success": True,
        "message": "Download link retrieved",
        "data": {
            "download_url": link["download_url"],
            "qr_code_url": link["qr_code_url"],
            "download_code": link["download_code"],
            "downloads_remaining": link["max_downloads"] - link["download_count"] - 1,
        },
    }


# ── Auto-Cleanup ──────────────────────────────────────────

def _schedule_cleanup(
    session_id: str,
    processing_id: str,
    photo_id: str,
    processed_path: str,
    qr_path: str,
):
    """Schedule deletion of all session data after CLEANUP_DELAY_SECONDS."""

    def _do_cleanup():
        time.sleep(CLEANUP_DELAY_SECONDS)
        try:
            db = get_supabase()

            # 1. Delete files from storage
            try:
                db.storage.from_("processed").remove([processed_path])
            except Exception:
                pass
            try:
                db.storage.from_("qrcodes").remove([qr_path])
            except Exception:
                pass

            # Delete original photos + preview
            photos_result = db.table("photos").select("original_path").eq("session_id", session_id).execute()
            for p in photos_result.data or []:
                try:
                    db.storage.from_("photos").remove([p["original_path"]])
                except Exception:
                    pass
            # Delete preview
            try:
                db.storage.from_("photos").remove([f"{session_id}/preview_nobg.png"])
            except Exception:
                pass

            # 2. Delete database records (cascading from session)
            db.table("download_links").delete().eq("processing_id", processing_id).execute()
            db.table("photo_processing").delete().eq("photo_id", photo_id).execute()
            db.table("face_expressions").delete().eq("photo_id", photo_id).execute()
            db.table("photos").delete().eq("session_id", session_id).execute()
            db.table("photo_sessions").update({"status": "expired"}).eq("id", session_id).execute()

            print(f"[Cleanup] Session {session_id} cleaned up after {CLEANUP_DELAY_SECONDS}s")
        except Exception as e:
            print(f"[Cleanup] Failed for session {session_id}: {e}")

    thread = threading.Thread(target=_do_cleanup, daemon=True)
    thread.start()
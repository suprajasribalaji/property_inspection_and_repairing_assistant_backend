import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, storage


def _adc_enabled() -> bool:
    v = os.environ.get("FIREBASE_USE_APPLICATION_DEFAULT", "").lower()
    return v in ("1", "true", "yes")


def _credential_from_env():
    """Build Admin credentials; prefer inline JSON env (no credentials file needed)."""
    json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if json_str:
        cred_dict = json.loads(json_str)
        return credentials.Certificate(cred_dict)

    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
        "FIREBASE_CREDENTIALS_PATH"
    )
    if path and os.path.isfile(path):
        return credentials.Certificate(path)

    if _adc_enabled():
        return credentials.ApplicationDefault()

    raise RuntimeError(
        "Set FIREBASE_SERVICE_ACCOUNT_JSON (recommended), or a credentials file path, "
        "or FIREBASE_USE_APPLICATION_DEFAULT=true for Application Default Credentials."
    )


def is_firebase_configured() -> bool:
    bucket = os.environ.get("FIREBASE_STORAGE_BUCKET", "").strip()
    if not bucket:
        return False

    if os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip():
        return True
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
        "FIREBASE_CREDENTIALS_PATH"
    )
    if path and os.path.isfile(path):
        return True
    if _adc_enabled():
        return True
    return False


def _ensure_app() -> None:
    if firebase_admin._apps:
        return
    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "").strip()
    if not bucket_name:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET is not set")

    cred = _credential_from_env()
    firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})


def _safe_filename(name: str | None) -> str:
    if not name:
        return "image"
    base = os.path.basename(name)
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", base)[:180] or "image"
    return base


def upload_inspection_image(
    image_bytes: bytes,
    mime_type: str | None,
    original_filename: str | None,
    session_id: str | None = None,
) -> dict | None:
    """
    Upload bytes to Firebase Storage. Returns dict with storage_path and download_url,
    or None if Firebase is not configured (env missing).
    """
    if not is_firebase_configured():
        return None

    _ensure_app()
    bucket = storage.bucket()
    
    # Use session_id if provided, otherwise generate UUID
    if session_id:
        folder = f"inspections/{session_id}"
    else:
        folder = f"inspections/{uuid.uuid4().hex}"
    
    fname = _safe_filename(original_filename)
    blob_path = f"{folder}/{fname}"

    blob = bucket.blob(blob_path)
    content_type = mime_type or "application/octet-stream"
    blob.upload_from_string(image_bytes, content_type=content_type)

    expires = datetime.now(timezone.utc) + timedelta(days=7)
    download_url = blob.generate_signed_url(expiration=expires, method="GET")

    return {
        "storage_path": blob_path,
        "download_url": download_url,
        "content_type": content_type,
    }

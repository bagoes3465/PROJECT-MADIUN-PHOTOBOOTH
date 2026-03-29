"""
AI Photobooth - APIFree.ai (Nano Banana Pro Edit) pipeline
"""
import io
import numpy as np
import cv2
from PIL import Image
from config import settings

# ── Configuration ──────────────────────────────────────────
MATTE_THRESHOLD = 18
MATTE_ERODE = 2
MATTE_DILATE = 1

LOCAL_PERSON_HEIGHT_RATIO = 0.78
LOCAL_MASCOT_HEIGHT_RATIO = 0.58
DEFAULT_CANVAS_HEIGHT = 1080
MIN_PERSON_ALPHA_COVERAGE = 0.015


# ── Background Removal ────────────────────────────────────

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background with GPU → CPU → simple fallback."""
    try:
        from rembg import remove
        result = remove(image)
        return _smooth_edges(result)
    except Exception:
        try:
            import os
            os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
            from rembg import remove
            result = remove(image)
            return _smooth_edges(result)
        except Exception:
            return _smooth_edges(_simple_bg_removal(image))


def _simple_bg_removal(image: Image.Image) -> Image.Image:
    """Fallback: HSV-based background removal."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    img_array = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    img_array = np.array(image)
    img_array[:, :, 3] = mask
    return Image.fromarray(img_array, "RGBA")


def _smooth_edges(image: Image.Image) -> Image.Image:
    """Gaussian blur on alpha channel for soft edges."""
    if image.mode != "RGBA":
        return image
    img_array = np.array(image)
    alpha = img_array[:, :, 3]
    # Tighten alpha matte first so we avoid gray fringes/halos.
    _, alpha = cv2.threshold(alpha, MATTE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.erode(alpha, kernel, iterations=MATTE_ERODE)
    alpha = cv2.dilate(alpha, kernel, iterations=MATTE_DILATE)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 1)
    img_array[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(img_array, "RGBA")


# ── APIFree.ai Pipeline (Flux 2 DEV Edit) ────────────────

def _is_apifree_enabled() -> bool:
    """Check if APIFree.ai API key is configured."""
    key = (settings.apifree_api_key or "").strip()
    return bool(key)


def _upload_temp_image_to_supabase(image: Image.Image) -> str:
    """Upload image to Supabase storage temporarily and return a public URL."""
    import uuid
    from database import get_supabase

    db = get_supabase()
    img_bytes = _image_to_png_bytes(image.convert("RGB"))
    storage_path = f"temp_ai/{uuid.uuid4().hex}.png"

    db.storage.from_("photos").upload(
        storage_path,
        img_bytes,
        {"content-type": "image/png"},
    )

    public_url = db.storage.from_("photos").get_public_url(storage_path)
    print(f"[AI] Uploaded temp image to Supabase: {storage_path}")
    return public_url


def _cleanup_temp_image(storage_path: str):
    """Remove temporary image from Supabase storage."""
    try:
        from database import get_supabase
        db = get_supabase()
        db.storage.from_("photos").remove([storage_path])
    except Exception:
        pass


def _run_apifree_cartoon_merge(
    composite_img: Image.Image,
    person_img: Image.Image,
    canvas_size: tuple[int, int],
    prompt_suffix: str = "",
    mascot_img: Image.Image | None = None,
) -> Image.Image:
    """Use APIFree.ai (Nano Banana Pro Edit) with 3 images: composite + face ref + mascot ref."""
    import time
    import requests as req

    api_key = (settings.apifree_api_key or "").strip()
    if not api_key:
        raise RuntimeError("APIFREE_API_KEY belum dikonfigurasi.")

    model_name = "google/nano-banana-pro/edit"
    base_url = (settings.apifree_base_url or "https://api.apifree.ai").strip().rstrip("/")
    timeout = max(60, int(settings.apifree_timeout_seconds or 300))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Upload 3 images: composite + person face ref + mascot identity ref
    print("[AI] Uploading 3 images to Supabase...")
    composite_url = _upload_temp_image_to_supabase(composite_img)
    person_url = _upload_temp_image_to_supabase(person_img)
    image_urls = [composite_url, person_url]
    if mascot_img is not None:
        mascot_url = _upload_temp_image_to_supabase(mascot_img.convert("RGB"))
        image_urls.append(mascot_url)

    prompt = _build_apifree_prompt(prompt_suffix, has_mascot_ref=(mascot_img is not None))

    # Determine aspect ratio from canvas size
    cw, ch = canvas_size
    from math import gcd
    g = gcd(cw, ch)
    aspect_ratio = f"{cw // g}:{ch // g}"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "image_urls": image_urls,
        "aspect_ratio": aspect_ratio,
        "resolution": "1K",
    }

    print(f"[AI] APIFree.ai submitting {len(image_urls)} images model={model_name} aspect={aspect_ratio}")

    # 1. Submit request
    resp = req.post(f"{base_url}/v1/image/submit", headers=headers, json=payload, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"APIFree.ai submit error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    if data.get("code") != 200:
        error = data.get("error", data.get("code_msg", "Unknown error"))
        raise RuntimeError(f"APIFree.ai submit error: {error}")

    request_id = data.get("resp_data", {}).get("request_id")
    if not request_id:
        raise RuntimeError(f"APIFree.ai did not return request_id: {data}")

    print(f"[AI] APIFree.ai submitted. request_id={request_id}")

    # 2. Poll for result
    max_polls = 60  # Max ~2 minutes (2s interval)
    for poll_num in range(max_polls):
        time.sleep(2)

        check_url = f"{base_url}/v1/image/{request_id}/result"
        check_resp = req.get(check_url, headers=headers, timeout=30)
        check_data = check_resp.json()

        if check_data.get("code") != 200:
            code_msg = check_data.get("code_msg", "Unknown")
            print(f"[AI] APIFree.ai poll error: {code_msg}")
            continue

        status = check_data.get("resp_data", {}).get("status", "")

        if status == "success":
            image_list = check_data.get("resp_data", {}).get("image_list", [])
            if not image_list:
                raise RuntimeError("APIFree.ai success tapi image_list kosong.")

            # Download the generated image
            img_url = image_list[0]
            print(f"[AI] APIFree.ai success! Downloading result...")
            img_resp = req.get(img_url, timeout=60)
            if img_resp.status_code != 200:
                raise RuntimeError(f"APIFree.ai gagal download image: {img_resp.status_code}")

            result_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            print(f"[AI] APIFree.ai done model={model_name} output_size={result_img.size}")
            return _fit_image_to_canvas(result_img, canvas_size)

        elif status in ("error", "failed"):
            error_msg = check_data.get("resp_data", {}).get("error", "Unknown error")
            raise RuntimeError(f"APIFree.ai task failed: {error_msg}")

        # Still processing
        if poll_num % 5 == 0:
            print(f"[AI] APIFree.ai polling... status={status} ({poll_num * 2}s)")

    raise RuntimeError("APIFree.ai timeout - task tidak selesai dalam waktu yang ditentukan.")


def _build_apifree_prompt(prompt_suffix: str = "", has_mascot_ref: bool = False) -> str:
    """Prompt for Nano Banana Pro Edit: 3 images input."""
    if has_mascot_ref:
        image_desc = (
            "I am providing 3 images. "
            "Image 1 is the MAIN SCENE: a photobooth composite with a background, a mascot character on the left, and a real person on the right. "
            "Image 2 is the PERSON FACE REFERENCE: the original unedited photo of the person — use this to preserve the person's face exactly. "
            "Image 3 is the MASCOT IDENTITY REFERENCE: the original mascot character — use this to preserve the mascot's exact design and appearance. "
        )
    else:
        image_desc = (
            "I am providing 2 images. "
            "Image 1 is the MAIN SCENE: a photobooth composite with a background, a mascot character on the left, and a real person on the right. "
            "Image 2 is the PERSON FACE REFERENCE: the original unedited photo of the person. "
        )

    prompt = (
        image_desc +

        "Task: Enhance Image 1 into a premium cinematic photobooth photo where the mascot and the person are POSING TOGETHER like friends. "

        "MASCOT INTERACTION RULE — KEY GOAL: "
        "Make the mascot and the person interact naturally like they are best friends posing for a photo together. "
        "The mascot should look lively and engaged — for example: giving a thumbs up, waving, pointing at the person, "
        "doing a peace sign, high-fiving, or standing shoulder-to-shoulder with the person. "
        "The person and mascot should look like they are having fun together at a festive event. "
        "Adjust the mascot's pose slightly to create a natural friendly interaction. "

        "MASCOT IDENTITY RULE — HIGHEST PRIORITY: "
        "The mascot is a specific pre-designed cartoon character. "
        + ("Image 3 shows the EXACT original mascot design — the output mascot MUST keep the same identity as Image 3. " if has_mascot_ref else "") +
        "Keep the mascot's IDENTITY: same character design, same colors, same outfit, same art style. "
        "You may adjust the mascot's POSE slightly for interaction, but do NOT change its identity or design. "
        "Do NOT add realistic texture, plush, fur, felt, or fabric. Keep the cartoon art style. "

        "CRITICAL FACE RULE: "
        "The person's face MUST match Image 2 exactly. "
        "Do NOT generate a new face. Keep exact same features, skin tone, hairstyle, glasses. "

        "LAYOUT RULE: "
        "BOTH the person AND the mascot MUST be shown FULL BODY from head to feet. "
        "Do NOT crop or zoom. Keep mascot on left, person on right. "
        "The mascot and person should be close together, not far apart. "

        "CLOTHING RULE: Keep the person's clothing exactly as is. "

        "Allowed enhancements: cinematic lighting, soft glowing particles, "
        "golden sparkles, festive atmosphere, vibrant color grading, "
        "natural shadows, soft depth of field, confetti, bokeh. "

        "Style: professional event photobooth photography, ultra detailed, fun and lively. "

        "Do NOT add text, watermark, logo. "
        "Do NOT add extra people or characters."
    )

    if prompt_suffix:
        prompt += f" {prompt_suffix}"

    return prompt


def _build_negative_prompt() -> str:
    return (
        "different person, new face, altered face, different clothes, different outfit, "
        "different hairstyle, distorted anatomy, extra arms, extra legs, "
        "changed clothing texture, painted clothing, canvas texture on clothes, "

        "extra character, extra person, mascot, cartoon character, puppet, doll, "

        "cropped body, half body, close up, zoomed in, "

        "blurry, low quality, bad lighting, "

        "text, watermark, logo, title, written words"
    )


def _ensure_person_visibility(person_rgba: Image.Image, original_person: Image.Image) -> Image.Image:
    """If alpha matte fails and removes too much of the person, fallback to opaque person image."""
    if person_rgba.mode != "RGBA":
        person_rgba = person_rgba.convert("RGBA")

    alpha = np.array(person_rgba.split()[-1])
    coverage = float(np.count_nonzero(alpha)) / float(max(1, alpha.size))
    if coverage >= MIN_PERSON_ALPHA_COVERAGE:
        return person_rgba

    return original_person.convert("RGBA")


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int] | None:
    try:
        width_text, height_text = str(aspect_ratio).split(":", 1)
        width = int(width_text)
        height = int(height_text)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        return None
    return None


def _fit_background_to_ratio(image: Image.Image, aspect_ratio: str) -> Image.Image:
    ratio = _parse_aspect_ratio(aspect_ratio)
    if ratio is None:
        return image.convert("RGB")

    src = image.convert("RGB")
    src_w, src_h = src.size
    target_w, target_h = ratio
    target_ratio = target_w / target_h
    current_ratio = src_w / max(1, src_h)

    if abs(current_ratio - target_ratio) < 0.01:
        return src

    if current_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = max(0, (src_w - new_w) // 2)
        return src.crop((left, 0, left + new_w, src_h))

    new_h = int(src_w / target_ratio)
    top = max(0, (src_h - new_h) // 2)
    return src.crop((0, top, src_w, top + new_h))


def _resolve_canvas_size(aspect_ratio: str, fallback_size: tuple[int, int]) -> tuple[int, int]:
    ratio = _parse_aspect_ratio(aspect_ratio)
    if ratio is None:
        return fallback_size

    ratio_w, ratio_h = ratio
    target_h = DEFAULT_CANVAS_HEIGHT
    target_w = max(1, int(round(target_h * (ratio_w / ratio_h))))
    return target_w, target_h


def _compose_reference_scene(
    background_image: Image.Image,
    mascot_image: Image.Image,
    person_image: Image.Image,
    aspect_ratio: str = "2:3",
) -> Image.Image:
    canvas_size = _resolve_canvas_size(aspect_ratio, background_image.size)
    bg = _fit_image_to_canvas(_fit_background_to_ratio(background_image, aspect_ratio), canvas_size).convert("RGBA")
    bg_w, bg_h = bg.size

    person_rgba = remove_background(person_image).convert("RGBA")
    person_rgba = _ensure_person_visibility(person_rgba, person_image)
    person_rgba = _resize_by_height(person_rgba, int(bg_h * LOCAL_PERSON_HEIGHT_RATIO))

    mascot_rgba = mascot_image.convert("RGBA")
    mascot_rgba = _resize_by_height(mascot_rgba, int(bg_h * LOCAL_MASCOT_HEIGHT_RATIO))

    # Keep both subjects fully visible by scaling down together if horizontal span is too large.
    left_margin = int(bg_w * 0.05)
    right_margin = int(bg_w * 0.05)
    min_gap = int(bg_w * 0.02)
    max_span = max(1, bg_w - left_margin - right_margin)
    current_span = mascot_rgba.width + person_rgba.width + min_gap
    if current_span > max_span:
        scale = max_span / float(current_span)
        person_rgba = person_rgba.resize(
            (max(1, int(person_rgba.width * scale)), max(1, int(person_rgba.height * scale))),
            Image.Resampling.LANCZOS,
        )
        mascot_rgba = mascot_rgba.resize(
            (max(1, int(mascot_rgba.width * scale)), max(1, int(mascot_rgba.height * scale))),
            Image.Resampling.LANCZOS,
        )

    person_x = bg_w - person_rgba.width - right_margin
    mascot_x = left_margin
    if mascot_x + mascot_rgba.width + min_gap > person_x:
        mascot_x = max(0, person_x - min_gap - mascot_rgba.width)

    person_x = max(0, min(person_x, bg_w - person_rgba.width))
    mascot_x = max(0, min(mascot_x, bg_w - mascot_rgba.width))

    mascot_y = max(0, bg_h - mascot_rgba.height - int(bg_h * 0.06))
    person_y = max(0, bg_h - person_rgba.height - int(bg_h * 0.04))

    bg = _add_ground_shadow(bg, mascot_x, mascot_y, mascot_rgba.width, mascot_rgba.height)
    bg.alpha_composite(mascot_rgba, (mascot_x, mascot_y))
    bg = _add_ground_shadow(bg, person_x, person_y, person_rgba.width, person_rgba.height)
    bg.alpha_composite(person_rgba, (person_x, person_y))
    return bg.convert("RGB")


def _compose_scene_without_mascot(
    background_image: Image.Image,
    person_image: Image.Image,
    aspect_ratio: str = "2:3",
) -> Image.Image:
    """Compose background + person only (no mascot). Leave space on the left for mascot overlay later."""
    canvas_size = _resolve_canvas_size(aspect_ratio, background_image.size)
    bg = _fit_image_to_canvas(_fit_background_to_ratio(background_image, aspect_ratio), canvas_size).convert("RGBA")
    bg_w, bg_h = bg.size

    person_rgba = remove_background(person_image).convert("RGBA")
    person_rgba = _ensure_person_visibility(person_rgba, person_image)
    person_rgba = _resize_by_height(person_rgba, int(bg_h * LOCAL_PERSON_HEIGHT_RATIO))

    right_margin = int(bg_w * 0.05)
    person_x = bg_w - person_rgba.width - right_margin
    person_x = max(0, min(person_x, bg_w - person_rgba.width))
    person_y = max(0, bg_h - person_rgba.height - int(bg_h * 0.04))

    bg = _add_ground_shadow(bg, person_x, person_y, person_rgba.width, person_rgba.height)
    bg.alpha_composite(person_rgba, (person_x, person_y))
    return bg.convert("RGB")


def _overlay_mascot_on_result(
    ai_result: Image.Image,
    mascot_image: Image.Image,
    aspect_ratio: str = "2:3",
) -> Image.Image:
    """Overlay the original mascot on the AI-enhanced result with color/lighting matching."""
    bg = ai_result.convert("RGBA")
    bg_w, bg_h = bg.size

    mascot_rgba = mascot_image.convert("RGBA")
    mascot_rgba = _resize_by_height(mascot_rgba, int(bg_h * LOCAL_MASCOT_HEIGHT_RATIO))

    left_margin = int(bg_w * 0.05)
    mascot_x = left_margin
    mascot_x = max(0, min(mascot_x, bg_w - mascot_rgba.width))
    mascot_y = max(0, bg_h - mascot_rgba.height - int(bg_h * 0.06))

    # Color-match mascot to the AI result's ambient lighting
    mascot_matched = _match_lighting(mascot_rgba, bg, mascot_x, mascot_y)

    bg = _add_ground_shadow(bg, mascot_x, mascot_y, mascot_matched.width, mascot_matched.height)
    bg.alpha_composite(mascot_matched, (mascot_x, mascot_y))

    # Add subtle glow behind the mascot so it blends into the scene
    bg = _add_ambient_glow(bg, mascot_x, mascot_y, mascot_matched.width, mascot_matched.height)

    return bg.convert("RGB")


def _fit_image_to_canvas(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    src = image.convert("RGB")
    src_w, src_h = src.size
    if src_w == target_w and src_h == target_h:
        return src

    scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
    resized = src.resize(
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def _cartoonize_image(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    color = cv2.bilateralFilter(rgb, d=9, sigmaColor=80, sigmaSpace=80)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        5,
    )
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, edges_rgb)
    blended = cv2.addWeighted(cartoon, 0.72, color, 0.28, 0)
    return Image.fromarray(blended.astype(np.uint8), "RGB")


def _resize_by_height(image: Image.Image, target_height: int) -> Image.Image:
    target_height = max(1, target_height)
    ratio = target_height / max(1, image.height)
    return image.resize(
        (max(1, int(image.width * ratio)), target_height),
        Image.Resampling.LANCZOS,
    )


def _add_ground_shadow(canvas: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    shadow = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
    shadow_width = max(20, int(width * 0.72))
    shadow_height = max(10, int(height * 0.10))
    center_x = x + width // 2
    center_y = min(canvas.height - 1, y + height - shadow_height // 3)
    top_left = (max(0, center_x - shadow_width // 2), max(0, center_y - shadow_height // 2))
    bottom_right = (
        min(canvas.width - 1, center_x + shadow_width // 2),
        min(canvas.height - 1, center_y + shadow_height // 2),
    )

    cv2.ellipse(
        shadow,
        ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2),
        (max(1, (bottom_right[0] - top_left[0]) // 2), max(1, (bottom_right[1] - top_left[1]) // 2)),
        0,
        0,
        360,
        (20, 20, 20, 70),
        -1,
    )
    shadow[:, :, 3] = cv2.GaussianBlur(shadow[:, :, 3], (21, 21), 0)
    return Image.alpha_composite(canvas, Image.fromarray(shadow, "RGBA"))


def _local_cartoon_from_reference(reference_image: Image.Image) -> Image.Image:
    """Fallback stylization from one merged scene image."""
    scene = reference_image.convert("RGB")
    stylized = _cartoonize_image(scene)
    return _fit_image_to_canvas(stylized, scene.size)


def _match_lighting(
    mascot_rgba: Image.Image,
    bg_rgba: Image.Image,
    x: int,
    y: int,
) -> Image.Image:
    """Adjust mascot colors to match the ambient lighting of the background region."""
    mascot_arr = np.array(mascot_rgba).copy()
    alpha = mascot_arr[:, :, 3]

    # Sample the background region where the mascot will be placed
    bg_arr = np.array(bg_rgba)
    mh, mw = mascot_arr.shape[:2]
    bh, bw = bg_arr.shape[:2]

    # Clamp region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + mw)
    y2 = min(bh, y + mh)

    if x2 <= x1 or y2 <= y1:
        return mascot_rgba

    bg_region = bg_arr[y1:y2, x1:x2, :3].astype(np.float32)
    bg_mean = bg_region.mean(axis=(0, 1))  # Average RGB of background behind mascot

    # Neutral reference (128) — shift mascot colors toward the scene's ambient tone
    neutral = 128.0
    color_shift = (bg_mean - neutral) * 0.15  # Subtle 15% shift toward scene color

    # Apply shift only to non-transparent pixels
    mask = alpha > 0
    for c in range(3):
        channel = mascot_arr[:, :, c].astype(np.float32)
        channel[mask] = np.clip(channel[mask] + color_shift[c], 0, 255)
        mascot_arr[:, :, c] = channel.astype(np.uint8)

    # Slight brightness adjustment based on scene brightness
    bg_brightness = bg_mean.mean()
    mascot_rgb = mascot_arr[:, :, :3].astype(np.float32)
    mascot_brightness = mascot_rgb[mask].mean() if mask.any() else 128.0
    brightness_factor = 1.0 + (bg_brightness - mascot_brightness) / 512.0  # Very subtle
    brightness_factor = max(0.90, min(1.10, brightness_factor))

    for c in range(3):
        channel = mascot_arr[:, :, c].astype(np.float32)
        channel[mask] = np.clip(channel[mask] * brightness_factor, 0, 255)
        mascot_arr[:, :, c] = channel.astype(np.uint8)

    return Image.fromarray(mascot_arr, "RGBA")


def _add_ambient_glow(
    canvas: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Image.Image:
    """Add a subtle warm glow behind the mascot area to help it blend into the scene."""
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    glow = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
    center_x = x + width // 2
    center_y = y + height // 2
    radius_x = max(1, width // 2 + 20)
    radius_y = max(1, height // 2 + 20)

    cv2.ellipse(
        glow,
        (center_x, center_y),
        (radius_x, radius_y),
        0, 0, 360,
        (255, 220, 150, 18),  # Warm golden glow, very subtle
        -1,
    )
    glow[:, :, 3] = cv2.GaussianBlur(glow[:, :, 3], (51, 51), 0)

    glow_layer = Image.fromarray(glow, "RGBA")
    # Composite glow BEHIND mascot by compositing onto canvas first
    return Image.alpha_composite(canvas, glow_layer)


def _run_cartoon_merge(
    composite_img: Image.Image,
    person_img: Image.Image,
    canvas_size: tuple[int, int],
    prompt_suffix: str = "",
    mascot_img: Image.Image | None = None,
) -> Image.Image:
    """Send composite + face ref + mascot ref to APIFree.ai."""
    if not _is_apifree_enabled():
        raise RuntimeError("APIFREE_API_KEY belum dikonfigurasi.")

    return _run_apifree_cartoon_merge(composite_img, person_img, canvas_size, prompt_suffix, mascot_img)


# ── Main Pipeline ─────────────────────────────────────────

def process_photobooth(
    person_img: Image.Image,
    bg_img: Image.Image,
    mascot_img: Image.Image | None = None,
    filter_config: dict | None = None,
) -> tuple[Image.Image, bool]:
    """
    Create one final photobooth image with APIFree.ai enhancement.

    Flow:
    1. Local compositing to build a clean scene reference
    2. APIFree.ai image-to-image enhancement
    3. Local cartoon fallback if API fails
    """
    prompt_suffix = ""
    aspect_ratio = "2:3"
    if filter_config and isinstance(filter_config, dict):
        prompt_suffix = filter_config.get("prompt_suffix", "")
        aspect_ratio = str(filter_config.get("aspect_ratio") or "2:3")

    if mascot_img is None:
        raise RuntimeError("Mascot image is required for cartoon merge flow")

    canvas_size = _resolve_canvas_size(aspect_ratio, bg_img.size)

    # Stage 1: compose full scene (background + mascot + person)
    reference_scene = _compose_reference_scene(
        background_image=bg_img,
        mascot_image=mascot_img,
        person_image=person_img,
        aspect_ratio=aspect_ratio,
    )

    try:
        # Stage 2: AI enhance with 3 images (composite + face ref + mascot ref)
        ai_result = _run_cartoon_merge(
            composite_img=reference_scene,
            person_img=person_img,
            canvas_size=canvas_size,
            prompt_suffix=prompt_suffix,
            mascot_img=mascot_img,
        )
        return ai_result.convert("RGB"), True
    except Exception as error:
        print(f"[AI] using local cartoon fallback: {error}")
        # Fallback: compose locally then apply basic stylization
        reference_scene = _compose_reference_scene(
            background_image=bg_img,
            mascot_image=mascot_img,
            person_image=person_img,
            aspect_ratio=aspect_ratio,
        )
        fallback = _local_cartoon_from_reference(reference_scene)
        return fallback.convert("RGB"), False

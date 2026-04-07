"""
AI Photobooth - APIFree.ai (Nano Banana 2 Edit) pipeline
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


# ── APIFree.ai Pipeline (Nano Banana 2 Edit) ────────────────

def _is_apifree_enabled() -> bool:
    """Check if APIFree.ai API key is configured."""
    key = (settings.apifree_api_key or "").strip()
    return bool(key)


def _upload_temp_image_to_supabase(image: Image.Image, max_retries: int = 3) -> str:
    """Upload image to Supabase storage temporarily and return a public URL."""
    import uuid
    import time
    from database import get_supabase

    db = get_supabase()
    img_bytes = _image_to_png_bytes(image.convert("RGB"))
    storage_path = f"temp_ai/{uuid.uuid4().hex}.png"

    for attempt in range(max_retries):
        try:
            db.storage.from_("photos").upload(
                storage_path,
                img_bytes,
                {"content-type": "image/png"},
            )
            public_url = db.storage.from_("photos").get_public_url(storage_path)
            print(f"[AI] Uploaded temp image to Supabase: {storage_path}")
            return public_url
        except (ConnectionError, ConnectionResetError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"[AI] Upload retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


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
    """Use APIFree.ai (Nano Banana 2 Edit) with 3 images: composite + face ref + mascot ref."""
    import time
    import requests as req

    api_key = (settings.apifree_api_key or "").strip()
    if not api_key:
        raise RuntimeError("APIFREE_API_KEY belum dikonfigurasi.")

    model_name = settings.apifree_model
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
        "width": cw,
        "height": ch,
        "seed": 42,
        "guidance_scale": 12.0,
    }

    print(f"[AI] APIFree.ai submitting {len(image_urls)} images model={model_name} aspect={aspect_ratio}")

    # 1. Submit request (with retry for transient connection errors)
    resp = None
    for attempt in range(3):
        try:
            resp = req.post(f"{base_url}/v1/image/submit", headers=headers, json=payload, timeout=timeout)
            break
        except (ConnectionError, req.exceptions.ConnectionError, OSError) as e:
            if attempt < 2:
                wait = 3 * (attempt + 1)
                print(f"[AI] APIFree.ai submit retry {attempt + 1}/3 after {wait}s: {e}")
                import time
                time.sleep(wait)
            else:
                raise RuntimeError(f"APIFree.ai connection failed after 3 retries: {e}")

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
    """Build prompt for APIFree.ai edit model (max 2500 chars)."""
    if has_mascot_ref:
        image_desc = (
            "3 images. Image1: photobooth scene (background+mascot left+person right). "
            "Image2: person face reference. Image3: mascot identity reference. "
        )
    else:
        image_desc = (
            "2 images. Image1: photobooth scene (background+person on right side). "
            "Image2: person face reference. "
        )

    prompt = (
        image_desc +

        "Enhance Image1 into a professional photobooth photo. "

        "PRESERVE FROM IMAGE1: "
        "- Same background location, buildings, sky. "
        "- ALL people visible — if 2 people in Image1, keep 2 people. "
        "- Person face identical to Image2 (face shape, eyes, nose, lips, skin, hair, glasses). "
        "- Person body shape, clothing, shirt pattern, pants, shoes — exactly as Image1. "
        "- People must look like real photographic humans, not 3D cartoon. "

        "MASCOT: "
        + ("Match Image3 identity exactly including text/labels on mascot. " if has_mascot_ref else "") +
        "Keep mascot anatomy identical to Image3. Do NOT add or remove limbs. "
        "Render mascot as high-quality 3D CGI character with scene-matching lighting. "

        "COMPOSITION: Mascot left, people right, all full body head to feet, no cropping. "
        "Subjects well-lit, clearly visible. Gentle background blur, subtle depth of field. "
        "Natural lighting, soft ground shadows. "

        "No watermark or logo."
    )

    if prompt_suffix:
        prompt += f" Style: {prompt_suffix}"

    return prompt


def _build_negative_prompt() -> str:
    return (
        "different person, new face, altered face, different clothes, different outfit, "
        "different hairstyle, distorted anatomy, extra arms, extra legs, "
        "deformed hands, wrong number of fingers, bent limbs, stretched limbs, "
        "changed body shape, fatter body, thinner body, altered proportions, "
        "cartoon person, stylized person, painted skin, anime person, "
        "3D rendered person, pixar person, animated person, CGI person, "
        "distorted mascot, deformed mascot, broken mascot anatomy, "
        "extra limbs on mascot, extra legs on mascot, extra arms on mascot, "
        "changed clothing texture, painted clothing, canvas texture on clothes, "

        "removed person, missing person, puppet, doll, "

        "cropped body, half body, close up, zoomed in, "

        "sparkles, glitter, light particles, magical effects, golden glow, lens flare, "

        "different background, missing background, blank background, white background, "
        "changed background, replaced background, "

        "blurry, low quality, bad lighting, "

        "watermark, logo"
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

    # Apply strong depth-based bokeh to background (cinematic shallow DOF)
    subject_regions = [
        (mascot_x, mascot_y, mascot_rgba.width, mascot_rgba.height),
        (person_x, person_y, person_rgba.width, person_rgba.height),
    ]
    bg = _apply_depth_bokeh(bg, subject_regions, strength=2.5)

    # Apply golden hour warm color grading to the background
    bg_rgb = _apply_golden_hour_grading(bg.convert("RGB"), intensity=0.35)
    bg = bg_rgb.convert("RGBA")
    # Restore alpha
    bg.putalpha(255)

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

    # Apply strong depth-based bokeh (cinematic shallow DOF)
    subject_regions = [
        (person_x, person_y, person_rgba.width, person_rgba.height),
    ]
    bg = _apply_depth_bokeh(bg, subject_regions, strength=2.5)

    # Apply golden hour warm color grading
    bg_rgb = _apply_golden_hour_grading(bg.convert("RGB"), intensity=0.35)
    bg = bg_rgb.convert("RGBA")
    bg.putalpha(255)

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


def _apply_bokeh_blur(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Apply realistic lens-bokeh blur to a background image.

    Uses a large disc kernel to simulate an out-of-focus camera lens.
    `strength` controls intensity: 1.0 = standard, 2.0 = strong, 3.0 = very strong.
    """
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = img_arr.shape[:2]

    # Scale kernel size with image resolution and strength — much larger for cinematic bokeh
    base_radius = max(11, int(min(h, w) * 0.028 * strength))
    ksize = base_radius * 2 + 1

    # Create a circular (disc) kernel for realistic bokeh look
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    cv2.circle(kernel, (center, center), base_radius, 1.0, -1)
    kernel /= kernel.sum()

    # Apply disc blur (simulates real lens bokeh)
    blurred = cv2.filter2D(img_arr, -1, kernel)

    # Second pass with Gaussian for creamy smooth bokeh
    gauss_ksize = max(3, (base_radius * 2 // 3) * 2 + 1)
    blurred = cv2.GaussianBlur(blurred, (gauss_ksize, gauss_ksize), 0)

    # Third pass for extra creaminess at high strength
    if strength >= 2.0:
        extra_ksize = max(3, base_radius) | 1
        blurred = cv2.GaussianBlur(blurred, (extra_ksize, extra_ksize), 0)

    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), "RGB")


def _apply_depth_bokeh(
    canvas: Image.Image,
    subject_regions: list[tuple[int, int, int, int]],
    strength: float = 1.0,
) -> Image.Image:
    """Apply cinematic depth-of-field bokeh — strong blur on background, sharp subjects.

    Uses a vertical depth gradient (top = far/blurry, bottom = near/sharp) combined
    with subject masks for realistic shallow-DOF like f/1.4 DSLR portrait.
    """
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    canvas_arr = np.array(canvas)
    rgb = canvas_arr[:, :, :3].copy()
    alpha = canvas_arr[:, :, 3].copy()
    h, w = rgb.shape[:2]

    # ── Build depth-based focus mask ──
    # Vertical gradient: top of image = 0 (far, max blur), bottom = 1 (near, sharp)
    depth_gradient = np.linspace(0.0, 1.0, h, dtype=np.float32)
    depth_mask = np.tile(depth_gradient[:, np.newaxis], (1, w))

    # Make the gradient steeper — only the bottom ~35% of the image is "near"
    # This simulates a low-angle shot with shallow DOF
    depth_mask = np.clip((depth_mask - 0.4) / 0.6, 0.0, 1.0) ** 0.7

    # Create subject foreground mask
    fg_mask = np.zeros((h, w), dtype=np.float32)
    for (sx, sy, sw, sh) in subject_regions:
        # Generous padding around subjects to keep edges sharp
        pad_x = int(sw * 0.12)
        pad_y = int(sh * 0.06)
        x1 = max(0, sx - pad_x)
        y1 = max(0, sy - pad_y)
        x2 = min(w, sx + sw + pad_x)
        y2 = min(h, sy + sh + pad_y)
        fg_mask[y1:y2, x1:x2] = 1.0

    # Ground near subjects stays semi-sharp (bottom 10%)
    ground_line = int(h * 0.92)
    fg_mask[ground_line:, :] = np.maximum(fg_mask[ground_line:, :], 0.3)

    # Smooth the subject mask edges with a large blur for natural DOF falloff
    blur_size = max(71, int(min(h, w) * 0.12)) | 1
    fg_mask = cv2.GaussianBlur(fg_mask, (blur_size, blur_size), 0)

    # Combine: sharp where subject OR near ground
    focus_mask = np.maximum(fg_mask, depth_mask)

    # Generate two levels of bokeh for progressive blur
    bokeh_strong = np.array(_apply_bokeh_blur(Image.fromarray(rgb, "RGB"), strength * 1.2))
    bokeh_medium = np.array(_apply_bokeh_blur(Image.fromarray(rgb, "RGB"), strength * 0.6))

    # Multi-level blend: strong bokeh for very far areas, medium for mid-range
    focus_3ch = focus_mask[:, :, np.newaxis]
    mid_blend = (rgb.astype(np.float32) * 0.3 + bokeh_medium.astype(np.float32) * 0.7)
    blended = (
        rgb.astype(np.float32) * focus_3ch +
        mid_blend * np.clip(1.0 - focus_3ch, 0, 0.5) * 2.0 +
        bokeh_strong.astype(np.float32) * np.clip(0.5 - focus_3ch, 0, 0.5) * 2.0
    )

    result = canvas_arr.copy()
    result[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    result[:, :, 3] = alpha
    return Image.fromarray(result, "RGBA")


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


def _apply_golden_hour_grading(image: Image.Image, intensity: float = 0.35) -> Image.Image:
    """Apply golden hour / sunset warm color grading to the image.

    Shifts colors toward warm orange-gold tones like a sunset photo.
    intensity: 0.0 = no change, 1.0 = full golden overlay.
    """
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)

    # Warm shift: boost red/green channels, reduce blue slightly
    warm_tint = np.array([25.0, 12.0, -15.0]) * intensity
    img_arr += warm_tint

    # Add a golden overlay using multiply blend
    golden = np.array([255.0, 200.0, 120.0]) / 255.0  # Warm golden color
    img_arr = img_arr / 255.0
    golden_blend = img_arr * (1.0 - intensity * 0.25) + (img_arr * golden) * (intensity * 0.25)
    img_arr = golden_blend * 255.0

    # Slight contrast boost for cinematic feel
    mean = img_arr.mean()
    img_arr = (img_arr - mean) * (1.0 + intensity * 0.15) + mean

    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8), "RGB")


def _add_ground_shadow(canvas: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """Add dramatic ground shadow beneath a subject — long, directional like sunset light."""
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    shadow = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
    # Wider, longer shadow for dramatic cinematic look
    shadow_width = max(30, int(width * 0.90))
    shadow_height = max(15, int(height * 0.18))
    center_x = x + width // 2
    # Offset shadow slightly to the right (as if light comes from left/behind)
    shadow_offset_x = int(width * 0.08)
    center_y = min(canvas.height - 1, y + height - shadow_height // 4)

    top_left = (max(0, center_x - shadow_width // 2 + shadow_offset_x), max(0, center_y - shadow_height // 2))
    bottom_right = (
        min(canvas.width - 1, center_x + shadow_width // 2 + shadow_offset_x),
        min(canvas.height - 1, center_y + shadow_height // 2),
    )

    cv2.ellipse(
        shadow,
        ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2),
        (max(1, (bottom_right[0] - top_left[0]) // 2), max(1, (bottom_right[1] - top_left[1]) // 2)),
        0,
        0,
        360,
        (15, 12, 10, 110),  # Darker, warmer shadow
        -1,
    )
    shadow[:, :, 3] = cv2.GaussianBlur(shadow[:, :, 3], (31, 31), 0)
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
        (255, 230, 180, 8),  # Very subtle warm glow for blending only
        -1,
    )
    glow[:, :, 3] = cv2.GaussianBlur(glow[:, :, 3], (71, 71), 0)

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
    Raises RuntimeError if API fails.
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

    # Stage 2: AI enhance with 3 images (composite + face ref + mascot ref)
    ai_result = _run_cartoon_merge(
        composite_img=reference_scene,
        person_img=person_img,
        canvas_size=canvas_size,
        prompt_suffix=prompt_suffix,
        mascot_img=mascot_img,
    )
    return ai_result.convert("RGB"), True

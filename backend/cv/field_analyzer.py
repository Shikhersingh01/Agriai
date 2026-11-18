"""Drone and Satellite Vegetation Health Analysis Module for AgriAI.

This module processes visible spectrum (RGB) aerial imagery captured by drones,
satellites, or high-clearance field rigs. It calculates a Simplified Green
Vegetation Index (Green Leaf Index - GLI), computes zone statistics (healthy,
moderate, and stressed canopy percentages), and produces a color-coded heatmap
overlay (Green = Healthy, Yellow = Moderate, Red = Stressed) encoded as a base64 PNG.

Key Metrics:
- Green Leaf Index (GLI): (2*G - R - B) / (2*G + R + B)
- Healthy Zone: Dense chlorophyll, vigorous vegetative canopy
- Moderate Zone: Transitional canopy, slight nutrient or moisture stress
- Stressed Zone: Chlorosis, bare patches, disease foci, or water deficit
"""

import io
import base64
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def _decode_field_image(image_bytes: bytes) -> Tuple[Any, Any]:
    """Decode raw image bytes into BGR numpy array with dimension clamping.

    Args:
        image_bytes: Raw bytes from aerial drone/satellite image.

    Returns:
        Tuple of (bgr_uint8_array, (height, width)).

    Raises:
        ValueError: If image bytes cannot be decoded.
    """
    import numpy as np

    # 1. Attempt decoding with OpenCV
    try:
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            h, w = bgr.shape[:2]
            # Downsample if image is extremely high resolution (e.g. 20+ megapixel drone mosaic)
            max_dim = 1200
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                new_w, new_h = int(w * scale), int(h * scale)
                bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return bgr, bgr.shape[:2]
    except Exception as e_cv:
        logger.debug(f"OpenCV decode bypassed: {e_cv}")

    # 2. Fallback using PIL
    try:
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        rgb = np.array(pil_img, dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()  # Convert RGB to BGR
        return bgr, bgr.shape[:2]
    except Exception as e_pil:
        raise ValueError(f"Failed to decode image using OpenCV and PIL: {e_pil}")


def _encode_image_to_base64_png(bgr_image: Any) -> str:
    """Encode BGR numpy array to a base64 PNG string.

    Args:
        bgr_image: BGR uint8 numpy array.

    Returns:
        UTF-8 base64 encoded PNG string.
    """
    import numpy as np

    # Try OpenCV encoding
    try:
        import cv2
        success, buffer = cv2.imencode(".png", bgr_image)
        if success:
            return base64.b64encode(buffer).decode("utf-8")
    except Exception as e_cv:
        logger.debug(f"OpenCV PNG encode bypassed: {e_cv}")

    # Fallback to PIL
    from PIL import Image
    rgb = bgr_image[:, :, ::-1]  # BGR to RGB
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _draw_hud_legend(overlay: Any) -> Any:
    """Draw a clean semi-transparent legend bar at the bottom of the heatmap.

    Args:
        overlay: BGR image numpy array.

    Returns:
        Overlay image with legend banner rendered.
    """
    import numpy as np
    try:
        import cv2
        h, w = overlay.shape[:2]
        bar_h = max(28, int(h * 0.05))
        y0 = h - bar_h

        # Semi-transparent dark background
        sub_img = overlay[y0:h, 0:w]
        black_rect = np.zeros_like(sub_img)
        overlay[y0:h, 0:w] = cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 0)

        # Legend dots & text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, w / 1600.0)
        thickness = 1

        items = [
            ("Healthy", (35, 205, 35)),
            ("Moderate", (20, 220, 240)),
            ("Stressed", (20, 20, 230))
        ]

        spacing = w // 4
        for i, (label, color) in enumerate(items):
            cx = int(w * 0.12 + i * spacing)
            cy = y0 + bar_h // 2
            cv2.circle(overlay, (cx, cy), 6, color, -1)
            cv2.putText(overlay, label, (cx + 12, cy + 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return overlay
    except Exception:
        return overlay


def analyze_field_image(image_bytes: bytes) -> Dict[str, Any]:
    """Analyze aerial vegetation health from drone or satellite photograph.

    Computes the Green Leaf Index (GLI), segments field pixels into healthy,
    moderate, and stressed zones, generates a color-coded heatmap overlay
    (Green = Healthy, Yellow = Moderate, Red = Stressed), and returns zone metrics.

    Args:
        image_bytes: Raw bytes from the uploaded aerial/field photograph.

    Returns:
        dict containing:
            - health_percentage: Float percentage of healthy vegetative canopy (0-100).
            - stress_percentage: Float percentage of stressed canopy or bare ground (0-100).
            - heatmap_base64: Base64-encoded PNG string of the color-coded overlay.
            - zone_analysis: Detailed dictionary containing breakdown of healthy,
                             moderate, and stressed zones, average vegetation index,
                             canopy status, and targeted agronomic recommendations.
    """
    if not image_bytes or not isinstance(image_bytes, bytes):
        return {
            "health_percentage": 0.0,
            "stress_percentage": 0.0,
            "heatmap_base64": "",
            "zone_analysis": {
                "healthy_percentage": 0.0,
                "moderate_percentage": 0.0,
                "stressed_percentage": 0.0,
                "error": "No image bytes provided."
            },
            "error": "Input image_bytes is empty or invalid."
        }

    try:
        import numpy as np

        # 1. Decode field image to BGR
        bgr, (height, width) = _decode_field_image(image_bytes)
        total_pixels = float(height * width)

        if total_pixels <= 0:
            raise ValueError("Field image has zero dimensions.")

        # 2. Extract color channels in float32
        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)

        # 3. Compute Simplified Green Vegetation Index (Green Leaf Index - GLI)
        # Formula: GLI = (2*G - R - B) / (2*G + R + B + epsilon)
        # Range is [-1.0, +1.0]
        denom = (2.0 * g + r + b) + 1e-6
        numer = (2.0 * g - r - b)
        gli = numer / denom

        # Normalized Green Ratio: g / (r + g + b + 1e-6)
        total_rgb = r + g + b + 1e-6
        green_ratio = g / total_rgb

        # 4. Zone Classification
        # Healthy: Vigorous chlorophyll (GLI >= 0.10 and green_ratio >= 0.36)
        healthy_mask = (gli >= 0.10) & (green_ratio >= 0.36)

        # Stressed: Low vegetation index, bare soil, severe chlorosis, moisture stress (GLI < 0.02)
        stressed_mask = (gli < 0.02) | (green_ratio < 0.32)

        # Moderate: Intermediate zone (not strictly healthy and not strictly stressed)
        moderate_mask = ~(healthy_mask | stressed_mask)

        # 5. Percentages Calculation
        healthy_count = float(np.sum(healthy_mask))
        moderate_count = float(np.sum(moderate_mask))
        stressed_count = float(np.sum(stressed_mask))

        healthy_pct = round((healthy_count / total_pixels) * 100.0, 2)
        moderate_pct = round((moderate_count / total_pixels) * 100.0, 2)
        stressed_pct = round((stressed_count / total_pixels) * 100.0, 2)

        # Ensure sum aligns to 100.0% accounting for rounding
        health_percentage = healthy_pct
        stress_percentage = stressed_pct

        # 6. Generate Color-Coded Heatmap (Green=Healthy, Yellow=Moderate, Red=Stressed)
        # Normalize GLI into [0.0, 1.0] across relevant vegetation range [-0.04, 0.24]
        gli_clamped = np.clip((gli - (-0.04)) / (0.24 - (-0.04)), 0.0, 1.0)

        # Color ramps in BGR format:
        # Stressed color: Red (20, 20, 230)
        # Moderate color: Yellow (20, 220, 240)
        # Healthy color: Green (35, 205, 35)
        color_map = np.zeros((height, width, 3), dtype=np.float32)

        # Lower half: Red to Yellow
        mask_low = gli_clamped <= 0.5
        t_low = (gli_clamped[mask_low] / 0.5)[:, np.newaxis]
        c_red = np.array([20.0, 20.0, 230.0], dtype=np.float32)
        c_yellow = np.array([20.0, 220.0, 240.0], dtype=np.float32)
        color_map[mask_low] = c_red * (1.0 - t_low) + c_yellow * t_low

        # Upper half: Yellow to Green
        mask_high = gli_clamped > 0.5
        t_high = ((gli_clamped[mask_high] - 0.5) / 0.5)[:, np.newaxis]
        c_green = np.array([35.0, 205.0, 35.0], dtype=np.float32)
        color_map[mask_high] = c_yellow * (1.0 - t_high) + c_green * t_high

        color_map = np.clip(color_map, 0, 255).astype(np.uint8)

        # 7. Alpha Blend Overlay with Original Aerial Image
        # 55% original photo context + 45% heatmap coloration
        alpha = 0.45
        overlay = (bgr.astype(np.float32) * (1.0 - alpha) + color_map.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Render HUD legend
        overlay = _draw_hud_legend(overlay)

        # 8. Encode to Base64 PNG
        heatmap_base64 = _encode_image_to_base64_png(overlay)

        # 9. Health Status and Actionable Agronomic Insights
        mean_gli = float(np.mean(gli))
        if healthy_pct >= 65.0:
            health_status = "High Canopy Vigor (Optimal)"
            intervention = (
                "Field shows uniform, dense vegetative canopy. Maintain scheduled irrigation "
                "cycles and routine pest scouting."
            )
        elif healthy_pct >= 40.0 or moderate_pct >= 40.0:
            health_status = "Moderate Foliage Vigor"
            intervention = (
                "Transitional vegetative vigor detected. Check yellow zones for nitrogen deficiency, "
                "slight moisture stress, or weed competition. Recommend targeted booster fertilizer application."
            )
        else:
            health_status = "Severe Field Stress / Sparse Canopy"
            intervention = (
                "Critical stress zones detected in red sectors. Immediately investigate for root disease, "
                "salinity patches, localized drought, or crop lodging. Prioritize foliar spray and spot irrigation."
            )

        zone_analysis = {
            "healthy_percentage": healthy_pct,
            "moderate_percentage": moderate_pct,
            "stressed_percentage": stressed_pct,
            "total_pixels_analyzed": int(total_pixels),
            "average_gli_index": round(mean_gli, 4),
            "canopy_health_status": health_status,
            "recommended_action": intervention,
            "legend": {
                "green": "Healthy, vigorous canopy (GLI >= 0.10)",
                "yellow": "Moderate canopy, minor chlorosis or transition",
                "red": "Stressed vegetation, bare soil, or severe nutrient deficit"
            }
        }

        return {
            "health_percentage": health_percentage,
            "stress_percentage": stress_percentage,
            "heatmap_base64": heatmap_base64,
            "zone_analysis": zone_analysis
        }

    except Exception as exc:
        logger.error(f"Field analysis failed: {exc}", exc_info=True)
        return {
            "health_percentage": 0.0,
            "stress_percentage": 0.0,
            "heatmap_base64": "",
            "zone_analysis": {
                "healthy_percentage": 0.0,
                "moderate_percentage": 0.0,
                "stressed_percentage": 0.0,
                "error": str(exc)
            },
            "error": f"Aerial field analysis encountered an error: {str(exc)}"
        }

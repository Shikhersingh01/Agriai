"""Soil Quality and Nutrient Estimation Module for AgriAI.

This module analyzes soil photographs to evaluate physical color profiles,
estimate organic carbon and nutrient richness, and compute a normalized Soil
Quality Score (0-100) using OpenCV and NumPy histogram analysis.

Agronomic Basis:
- Dark brown / black hues indicate high organic carbon and humic matter.
- Light, pale, or bleached sand tones indicate nutrient depletion and low moisture retention.
- Reddish and orange hues indicate iron sesquioxides (lateritic soils) requiring phosphorus and pH management.
- Whitish or grayish hues indicate potential calcareous or saline crusts.
"""

import io
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def _decode_image_to_hsv(image_bytes: bytes) -> Tuple[Any, Any]:
    """Decode raw image bytes to RGB and HSV numpy arrays.

    Uses OpenCV if available; falls back smoothly to Pillow (PIL) and NumPy.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        Tuple of (bgr_or_rgb_array, hsv_array).

    Raises:
        ValueError: If image bytes cannot be decoded into an image.
    """
    import numpy as np

    # 1. Attempt decoding with OpenCV
    try:
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            return bgr, hsv
    except Exception as e_cv:
        logger.debug(f"OpenCV decode attempt bypassed: {e_cv}")

    # 2. Fallback to PIL
    try:
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)

        # Convert RGB to HSV using standard formulas
        # Normalize RGB to [0, 1]
        r = rgb[:, :, 0].astype(np.float32) / 255.0
        g = rgb[:, :, 1].astype(np.float32) / 255.0
        b = rgb[:, :, 2].astype(np.float32) / 255.0

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue calculation in degrees 0-360 mapped to OpenCV scale [0, 179]
        h = np.zeros_like(r)
        mask = delta > 1e-5

        # When cmax == r
        r_mask = mask & (cmax == r)
        h[r_mask] = (60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0))

        # When cmax == g
        g_mask = mask & (cmax == g)
        h[g_mask] = (60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0))

        # When cmax == b
        b_mask = mask & (cmax == b)
        h[b_mask] = (60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0))

        h_cv = (h / 2.0).astype(np.uint8)  # Map 0-360 to 0-179
        s_cv = np.where(cmax > 1e-5, (delta / cmax) * 255.0, 0).astype(np.uint8)
        v_cv = (cmax * 255.0).astype(np.uint8)

        hsv = np.stack([h_cv, s_cv, v_cv], axis=-1)
        return rgb, hsv
    except Exception as e_pil:
        raise ValueError(f"Failed to decode image using OpenCV and PIL: {e_pil}")


def analyze_soil_image(image_bytes: bytes) -> Dict[str, Any]:
    """Estimate soil quality and nutrient profile from a soil photograph.

    Converts the image to the HSV color space, calculates color channel histograms,
    estimates soil classification (organic dark, reddish iron-rich, sandy light,
    or alluvial loam), computes a composite soil quality score (0-100), and provides
    practical agricultural recommendations.

    Args:
        image_bytes: Raw bytes from the uploaded soil image file.

    Returns:
        dict containing:
            - quality_score: Computed integer score between 0 and 100.
            - quality_label: One of 'Excellent', 'Good', 'Fair', or 'Poor'.
            - color_profile: Dictionary containing dominant color, HSV metrics,
                             estimated organic matter, and soil classification.
            - recommendations: List of agronomic management steps.
    """
    if not image_bytes or not isinstance(image_bytes, bytes):
        return {
            "quality_score": 0,
            "quality_label": "Poor",
            "color_profile": {"error": "No image data provided"},
            "recommendations": ["Please upload a clear, focused photograph of soil surface."]
        }

    try:
        import numpy as np

        # 1. Decode image to HSV color space
        _, hsv = _decode_image_to_hsv(image_bytes)

        # Downsample if image is excessively large for speed
        h_dim, w_dim = hsv.shape[:2]
        if h_dim > 600 or w_dim > 600:
            step_y = max(1, h_dim // 400)
            step_x = max(1, w_dim // 400)
            hsv = hsv[::step_y, ::step_x]

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        total_pixels = int(hsv.shape[0] * hsv.shape[1])
        if total_pixels == 0:
            raise ValueError("Decoded image contains zero pixels.")

        # 2. Histogram Analysis via NumPy
        hist_h, bin_edges_h = np.histogram(h_channel, bins=18, range=(0, 180))
        hist_s, bin_edges_s = np.histogram(s_channel, bins=10, range=(0, 256))
        hist_v, bin_edges_v = np.histogram(v_channel, bins=10, range=(0, 256))

        # Mean and standard deviation of HSV channels
        mean_h = float(np.mean(h_channel))
        mean_s = float(np.mean(s_channel))
        mean_v = float(np.mean(v_channel))

        # 3. Categorize Soil Color Distributions
        # A. Dark Brown / Humic (Rich Organic Carbon)
        # Low Value (dark: V < 100), moderate Saturation (S >= 25), earth tones (8 <= H <= 32)
        # or very dark soil (V < 75)
        organic_mask = ((v_channel < 100) & (s_channel >= 25) & (h_channel >= 8) & (h_channel <= 32)) | (v_channel < 75)
        p_organic = float(np.sum(organic_mask) / total_pixels)

        # B. Reddish / Iron-Rich Lateritic
        # Red/orange hues (H <= 12 or H >= 165), distinct saturation (S >= 55), moderate brightness (50 <= V <= 190)
        red_mask = ((h_channel <= 12) | (h_channel >= 165)) & (s_channel >= 55) & (v_channel >= 50) & (v_channel <= 190)
        p_red = float(np.sum(red_mask) / total_pixels)

        # C. Sandy / Light / Pale Loam (Low Organic Matter)
        # High brightness (V >= 140), low to moderate saturation (S <= 75)
        sandy_mask = (v_channel >= 140) & (s_channel <= 75)
        p_sandy = float(np.sum(sandy_mask) / total_pixels)

        # D. Alluvial Brown Silt Loam
        # Moderate hue (12 <= H <= 30), balanced saturation (30 <= S <= 140), medium brightness (90 <= V <= 150)
        alluvial_mask = (h_channel >= 12) & (h_channel <= 30) & (s_channel >= 30) & (s_channel <= 140) & (v_channel >= 90) & (v_channel <= 150)
        p_alluvial = float(np.sum(alluvial_mask) / total_pixels)

        # E. Saline / Calcareous / Bleached Gray
        # Bright (V >= 165) and very low saturation (S < 25)
        saline_mask = (v_channel >= 165) & (s_channel < 25)
        p_saline = float(np.sum(saline_mask) / total_pixels)

        # 4. Compute Soil Quality Score (0 - 100)
        raw_score = (
            p_organic * 94.0 +
            p_alluvial * 82.0 +
            p_red * 70.0 +
            p_sandy * 42.0 +
            p_saline * 25.0
        )

        # If majority is neither classified, blend based on baseline brightness
        classified_sum = p_organic + p_alluvial + p_red + p_sandy + p_saline
        if classified_sum < 0.85:
            remainder = 1.0 - classified_sum
            # Baseline value contribution: darker is generally more fertile
            baseline = max(30.0, 90.0 - (mean_v / 255.0) * 55.0)
            raw_score += remainder * baseline

        # Penalties and bonuses
        if mean_v > 180:
            raw_score -= (mean_v - 180) * 0.35  # Extreme pale / bleached penalty
        elif mean_v < 85 and mean_s > 30:
            raw_score += 4.0  # Rich dark humus bonus

        quality_score = int(round(max(10.0, min(96.0, raw_score))))

        # 5. Quality Label
        if quality_score >= 80:
            quality_label = "Excellent"
        elif quality_score >= 65:
            quality_label = "Good"
        elif quality_score >= 45:
            quality_label = "Fair"
        else:
            quality_label = "Poor"

        # 6. Dominant Soil Type & Color Profile
        proportions = {
            "Organic Rich (Dark Brown / Black)": p_organic,
            "Alluvial Loam (Medium Brown)": p_alluvial,
            "Iron-Rich Laterite (Reddish)": p_red,
            "Sandy / Light Loam (Pale)": p_sandy,
            "Calcareous / Saline (Grayish-White)": p_saline
        }
        dominant_type = max(proportions, key=proportions.get)

        if p_organic >= 0.30 or dominant_type == "Organic Rich (Dark Brown / Black)":
            dominant_color = "Dark Brown / Black"
            soil_classification = "Humus-Rich / Black Cotton Soil (Vertisol/Mollisol)"
            organic_matter_estimate = "High (> 3.0%)"
        elif p_red >= 0.25 or dominant_type == "Iron-Rich Laterite (Reddish)":
            dominant_color = "Reddish-Brown / Terracotta"
            soil_classification = "Red Lateritic Loam (Rich in Iron Sesquioxides)"
            organic_matter_estimate = "Moderate (1.2% - 2.2%)"
        elif p_sandy >= 0.35 or dominant_type == "Sandy / Light Loam (Pale)":
            dominant_color = "Pale Sandy / Light Brown"
            soil_classification = "Light Sandy Loam / Arid Soil"
            organic_matter_estimate = "Low (< 1.0%)"
        elif p_saline >= 0.20 or dominant_type == "Calcareous / Saline (Grayish-White)":
            dominant_color = "Grayish-White / Chalky"
            soil_classification = "Calcareous / Potentially Saline Soil"
            organic_matter_estimate = "Very Low (< 0.6%)"
        else:
            dominant_color = "Medium Brown"
            soil_classification = "Alluvial Silt Loam (Balanced Indo-Gangetic Soil)"
            organic_matter_estimate = "Medium (1.5% - 2.5%)"

        color_profile = {
            "dominant_color": dominant_color,
            "soil_classification": soil_classification,
            "estimated_organic_matter": organic_matter_estimate,
            "mean_hsv": {
                "hue": round(mean_h, 1),
                "saturation": round(mean_s, 1),
                "value_brightness": round(mean_v, 1)
            },
            "channel_distributions_pct": {
                "organic_rich_dark": round(p_organic * 100, 1),
                "alluvial_brown": round(p_alluvial * 100, 1),
                "iron_rich_red": round(p_red * 100, 1),
                "sandy_light": round(p_sandy * 100, 1),
                "saline_calcareous": round(p_saline * 100, 1)
            },
            "histogram_summary": {
                "h_bins_count": len(hist_h),
                "s_bins_count": len(hist_s),
                "v_bins_count": len(hist_v)
            }
        }

        # 7. Agricultural Recommendations Based on Color Profile
        recommendations: List[str] = []

        if "Humus-Rich" in soil_classification or quality_score >= 80:
            recommendations.extend([
                "Excellent organic reserve: Soil has superior moisture retention and cation exchange capacity.",
                "Ideal for nutrient-demanding crops like Wheat, Sugarcane, Basmati Rice, and winter vegetables.",
                "Adopt minimum or conservation tillage to protect soil aggregate structure and humus layers.",
                "Use balanced NPK fertilization; avoid over-applying nitrogen to prevent crop lodging and vegetative surplus."
            ])
        elif "Red Lateritic" in soil_classification:
            recommendations.extend([
                "High iron and aluminum content: Typical of subtropical red lateritic tracts prone to phosphorus fixation.",
                "Soil testing for pH is strongly advised: Red soils are often acidic (pH 5.2 - 6.2). Apply agricultural lime or dolomite @ 1.5-2.0 tonnes/ha if pH < 6.0.",
                "Apply Single Superphosphate (SSP) or DAP banded close to root zones, supplemented with Phosphate Solubilizing Bacteria (PSB) biofertilizer.",
                "Incorporate 8-10 tonnes/ha of well-decomposed Farmyard Manure (FYM) or green manure (Dhaincha/Sunn hemp) to enhance nutrient buffering.",
                "Ideal for Groundnut, Maize, Cotton, Pulses (Arhar/Gram), and horticultural fruit orchards."
            ])
        elif "Sandy" in soil_classification:
            recommendations.extend([
                "Low organic carbon and light texture: High permeability leads to rapid drainage and nutrient leaching.",
                "Add well-rotted Farmyard Manure (FYM) @ 10-15 tonnes/ha or vermicompost @ 4 tonnes/ha before sowing.",
                "Divide nitrogen and potash fertilizers into 3-4 split applications to avoid leaching past the active root zone.",
                "Adopt micro-irrigation (drip or sprinkler) with organic mulching (paddy straw) to conserve soil moisture.",
                "Well suited for drought-tolerant crops: Pearl Millet (Bajra), Cluster Bean (Guar), Mustard, and Chickpea."
            ])
        elif "Calcareous / Potentially Saline" in soil_classification or quality_score < 45:
            recommendations.extend([
                "Light grayish crust detected: Potential indication of high free calcium carbonates or soluble salts.",
                "Conduct laboratory soil testing for Electrical Conductivity (ECe) and Exchangeable Sodium Percentage (ESP).",
                "If soil is sodic/alkaline (pH > 8.2), apply agricultural gypsum @ 2-4 tonnes/ha followed by fresh water flushing.",
                "Plant salt-tolerant cultivars such as KRL-210 (Wheat), CSR-36 (Paddy), or CS-54 (Mustard).",
                "Avoid saline borewell water; incorporate organic residues to increase soil microbial activity."
            ])
        else:  # Alluvial Loam
            recommendations.extend([
                "Balanced alluvial texture: Favorable water retention, porosity, and root aeration for Indo-Gangetic plains.",
                "Replenish micronutrients (Zinc Sulfate 21% @ 25 kg/ha and Boron @ 5 kg/ha) following intense wheat-paddy rotation.",
                "Incorporate green manuring (Sesbania) in the summer fallow window to maintain organic carbon above 0.75%.",
                "Follow Recommended Dose of Fertilizers (RDF) based on localized Soil Health Card targets."
            ])

        return {
            "quality_score": quality_score,
            "quality_label": quality_label,
            "color_profile": color_profile,
            "recommendations": recommendations
        }

    except Exception as exc:
        logger.error(f"Soil analysis failed: {exc}", exc_info=True)
        return {
            "quality_score": 50,
            "quality_label": "Fair",
            "color_profile": {"error": str(exc)},
            "recommendations": [
                "Analysis encountered an unexpected format error.",
                "Ensure the image depicts clean, clear soil under natural diffused lighting without shadows."
            ]
        }

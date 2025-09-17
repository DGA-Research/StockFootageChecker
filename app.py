import os
import io
import shutil
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import imagehash
import cv2  # Fallback frame extraction without system ffmpeg

# PySceneDetect imports
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Google Cloud Vision
from google.cloud import vision
from google.oauth2 import service_account

import numpy as np


# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_1fps_opencv(infile: Path, outdir: Path) -> List[Path]:
    """Fallback: extract ~1 fps frames using OpenCV (no system ffmpeg needed)."""
    ensure_dir(outdir)
    cap = cv2.VideoCapture(str(infile))
    if not cap.isOpened():
        st.error("OpenCV could not open the video. Install system ffmpeg or try a different codec.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    step = int(round(fps))

    frame_idx = 0
    saved_paths: List[Path] = []

    ok, frame = cap.read()
    while ok:
        if frame_idx % max(step, 1) == 0:
            out_path = outdir / f"frame_{frame_idx:06d}.jpg"
            try:
                cv2.imwrite(str(out_path), frame)
                saved_paths.append(out_path)
            except Exception:
                pass
        ok, frame = cap.read()
        frame_idx += 1

    cap.release()
    return saved_paths


def scenes_to_keyframes(video_path: Path, outdir: Path, threshold: float, min_scene_len: int) -> List[Tuple[Path, Tuple[int, int]]]:
    ensure_dir(outdir)
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()

    results: List[Tuple[Path, Tuple[int, int]]] = []
    for i, (start, end) in enumerate(scene_list):
        start_f, end_f = start.get_frames(), end.get_frames()
        mid = start_f + (end_f - start_f) // 2
        frame_path = outdir / f"scene_{i:04d}.jpg"
        try:
            video.save_frame(mid, str(frame_path))
            results.append((frame_path, (start_f, end_f)))
        except Exception:
            continue
    return results


def dedupe_frames_by_phash(frame_paths: List[Path], max_distance: int = 5) -> List[Path]:
    kept: List[Path] = []
    hashes: List[Tuple[Path, imagehash.ImageHash]] = []
    for p in frame_paths:
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im)
        except Exception:
            continue
        is_dup = False
        for _, existing in hashes:
            if h - existing <= max_distance:
                is_dup = True
                break
        if not is_dup:
            kept.append(p)
            hashes.append((p, h))
    return kept


# ---- Credential bootstrap (secrets first, then env var) ----

def get_vision_client():
    """Return a Vision client using Streamlit secrets if available, else env var. None if neither is set."""
    try:
        if "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]))
            return vision.ImageAnnotatorClient(credentials=creds)
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # Uses default credentials loader which reads the env var path
            return vision.ImageAnnotatorClient()
    except Exception:
        return None
    return None

VISION_CLIENT = get_vision_client()


def gcv_web_detection_for_image_bytes(img_bytes: bytes) -> Dict[str, Any]:
    """Call Google Vision Web Detection for a single image (bytes)."""
    if VISION_CLIENT is None:
        raise RuntimeError(
            "Google Vision credentials not configured: add `gcp_service_account` "
            "to st.secrets or set GOOGLE_APPLICATION_CREDENTIALS."
        )
    image = vision.Image(content=img_bytes)
    resp = VISION_CLIENT.web_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    wd = resp.web_detection
    return {
        "web_entities": [
            {"description": e.description, "score": getattr(e, "score", None)}
            for e in getattr(wd, "web_entities", [])
            if getattr(e, "description", None)
        ],
        "visually_similar": [
            getattr(i, "url", None)
            for i in getattr(wd, "visually_similar_images", [])
            if getattr(i, "url", None)
        ],
        "pages_with_matches": [
            getattr(p, "url", None)
            for p in getattr(wd, "pages_with_matching_images", [])
            if getattr(p, "url", None)
        ],
    }


def risk_tags_from_metadata(urls: List[str], entities: List[Dict[str, Any]]) -> List[str]:
    tags = set()
    joined = " ".join([u.lower() for u in urls])
    if any(x in joined for x in ["/editorial", "editorial", "/news/"]):
        tags.add("Editorial-only?")
    if any(x in joined for x in ["gettyimages.com", "istockphoto.com", "shutterstock.com", "adobe.com/stock"]):
        tags.add("Likely stock source")
    sensitive = {"protest", "police", "covid", "russia", "war", "military", "donald trump", "joe biden"}
    ent_text = " ".join([str(e.get("description", "")).lower() for e in entities])
    if any(s in ent_text for s in sensitive):
        tags.add("Sensitive context")
    return sorted(tags)

def _mask_text_with_vision(img_bgr: 'np.ndarray') -> 'np.ndarray':
    """Use Google Vision TEXT_DETECTION to find large text regions and black them out.
    Returns a copy of the image with masks applied. If Vision client is missing or fails, returns original.
    """
    try:
        if VISION_CLIENT is None:
            return img_bgr
        # Encode to PNG bytes for Vision
        ok, buf = cv2.imencode('.png', img_bgr)
        if not ok:
            return img_bgr
        img_bytes = buf.tobytes()
        image = vision.Image(content=img_bytes)
        resp = VISION_CLIENT.text_detection(image=image)
        if getattr(resp, 'error', None) and resp.error.message:
            return img_bgr
        ann = resp.text_annotations
        if not ann:
            return img_bgr
        # First element is full text; subsequent are words with bounding polys
        out = img_bgr.copy()
        h, w = out.shape[:2]
        for word in ann[1:]:
            verts = [(v.x, v.y) for v in word.bounding_poly.vertices]
            xs = [x for x, y in verts if x is not None]
            ys = [y for x, y in verts if y is not None]
            if not xs or not ys:
                continue
            x1, y1, x2, y2 = max(min(xs), 0), max(min(ys), 0), min(max(xs), w-1), min(max(ys), h-1)
            # Heuristic: mask only larger text boxes (> ~3% width, > ~1.5% height)
            if (x2 - x1) / max(w,1) >= 0.03 and (y2 - y1) / max(h,1) >= 0.015:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        return out
    except Exception:
        return img_bgr


def _unsharp_mask(img_bgr: 'np.ndarray', amount: float = 0.6) -> 'np.ndarray':
    """Simple unsharp mask for mild sharpening. amount in [0, ~1.5]."""
    try:
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    except Exception:
        return img_bgr


def preprocess_for_vision(path: Path, *, target_width: int = 960,
                          mask_overlay_text: bool = True, sharpen_amount: float = 0.6) -> bytes:
    """Load with OpenCV, optionally mask overlay text via Vision OCR,
    resize to target_width, optionally sharpen; return PNG bytes."""
    img = cv2.imread(str(path))
    if img is None:
        with open(path, 'rb') as f:
            return f.read()
    h, w = img.shape[:2]
    if mask_overlay_text:
        img = _mask_text_with_vision(img)
    if w > 0 and target_width > 0 and w != target_width:
        new_h = max(1, int(h * (target_width / float(w))))
        img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    if sharpen_amount and sharpen_amount > 0:
        img = _unsharp_mask(img, amount=float(sharpen_amount))
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        with open(path, 'rb') as f:
            return f.read()
    return buf.tobytes()


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Stock Footage Identifier — MVP", layout="wide")
st.title("Stock Footage Identifier — MVP")

with st.sidebar:
    st.header("Analysis Settings")
    use_scene_detect = st.checkbox("Use scene detection (PySceneDetect)", value=True)
    threshold = st.slider("Scene threshold", 1.0, 60.0, 27.0, 1.0)
    min_scene_len = st.slider("Min scene length (frames)", 1, 60, 12, 1)
    max_frames = st.slider("Max frames to query", 1, 60, 12, 1)
    dedupe_hamming = st.slider("De-dup Hamming distance", 1, 16, 5, 1)
    blur_thresh = st.slider("Blur filter (Laplacian var)", 0.0, 500.0, 50.0, 5.0,
                            help="Filter out frames blurrier than this value. Higher = sharper only.")
    stock_only = st.checkbox(
        "Show stock domains only in results", value=True,
        help="Filter to common stock providers (Adobe, Getty/iStock, Shutterstock, Alamy, Pond5, Storyblocks, Depositphotos, Dreamstime). Others appear in 'other_urls'."
    )
    st.divider()
    st.subheader("Preprocessing")
    mask_overlay_text = st.checkbox(
        "Mask overlay text (OCR)", value=True,
        help="Detect large on-screen text (subtitles/banners) and black it out before reverse search."
    )
    target_width = st.slider(
        "Resize width for Vision (px)", 480, 1280, 960, 10,
        help="Normalize frames; 800–1000px often improves stock matches."
    )
    sharpen_amount = st.slider(
        "Sharpen amount (unsharp mask)", 0.0, 1.5, 0.6, 0.05,
        help="Mild sharpening can help Vision match stock previews."
    )
    st.caption("Tune sensitivity & cost. Use blur filter to avoid motion-blurred frames that confuse reverse image search.")


uploaded = st.file_uploader("Upload ad video (MP4/WEBM/MOV)", type=["mp4", "mov", "webm"]) 

if uploaded:
    workdir = Path(tempfile.mkdtemp(prefix="adscan_"))
    frames_dir = workdir / "frames"
    ensure_dir(frames_dir)

    video_path = workdir / uploaded.name
    with open(video_path, "wb") as f:
        f.write(uploaded.read())

    st.info(f"Saved to {video_path}")

    # 1) Extract keyframes
    with st.spinner("Detecting scenes / extracting frames..."):
        frame_records: List[Tuple[Path, Tuple[int, int]]] = []
        if use_scene_detect:
            try:
                frame_records = scenes_to_keyframes(video_path, frames_dir, threshold=float(threshold), min_scene_len=int(min_scene_len))
            except Exception as e:
                st.warning(f"Scene detection failed ({e}). Falling back to 1 fps.")
        if not frame_records:
            simple = extract_1fps_opencv(video_path, frames_dir)
            frame_records = [(p, (0, 0)) for p in simple]

    all_frames = [rec[0] for rec in frame_records]
    st.write(f"Extracted {len(all_frames)} raw frames.")

    # 2) De-duplicate
    with st.spinner("De-duplicating frames by perceptual hash..."):
        kept_frames = dedupe_frames_by_phash(all_frames, max_distance=int(dedupe_hamming))
    st.write(f"Kept {len(kept_frames)} unique frames after de-dup.")

    # 3) Limit to max_frames (simple sample)
    # Filter out blurry frames using Laplacian variance
    sharp_frames: List[Path] = []
    for p in kept_frames:
        try:
            img = cv2.imread(str(p))
            if img is None:
                continue
            fm = cv2.Laplacian(img, cv2.CV_64F).var()
            if fm >= float(blur_thresh):
                sharp_frames.append(p)
        except Exception:
            # if anything goes wrong, keep the frame rather than drop
            sharp_frames.append(p)

    sample_frames = sharp_frames[: int(max_frames)]

    st.subheader("Sampled Frames")
    cols = st.columns(min(4, len(sample_frames)) or 1)
    for i, p in enumerate(sample_frames):
        with cols[i % len(cols)]:
            st.image(str(p), caption=p.name, width="stretch")

    # 4) Call Google Vision Web Detection
    st.subheader("Reverse Image Lookups (Google Vision Web Detection)")

    results: List[Dict[str, Any]] = []

    if VISION_CLIENT is None:
        st.warning(
            "Google Vision not configured. Add your service account JSON to `st.secrets['gcp_service_account']` "
            "or set the `GOOGLE_APPLICATION_CREDENTIALS` env var."
        )
    else:
        with st.spinner("Querying Vision API (Web Detection)..."):
            for p in sample_frames:
                try:
                    # Preprocess (mask overlay text, resize, sharpen) then PNG bytes
                    img_bytes = preprocess_for_vision(
                        p,
                        target_width=int(target_width),
                        mask_overlay_text=bool(mask_overlay_text),
                        sharpen_amount=float(sharpen_amount),
                    )


                    wd = gcv_web_detection_for_image_bytes(img_bytes)

                    # Collect URLs and prioritize stock providers
                    all_urls = list({*(wd.get("visually_similar", []) + wd.get("pages_with_matches", []))})
                    _stock_domains = (
                        "shutterstock.com", "adobe.com/stock", "stock.adobe.com",
                        "gettyimages.com", "istockphoto.com", "alamy.com",
                        "pond5.com", "storyblocks.com", "depositphotos.com",
                        "dreamstime.com", "123rf.com", "videoblocks.com",
                        "envato.com", "motionarray.com"
                    )
                    stock_urls = [u for u in all_urls if any(d in u for d in _stock_domains)]
                    other_urls = [u for u in all_urls if u not in stock_urls]
                    urls = stock_urls if stock_only else all_urls # prefer stock when present

                    entities = wd.get("web_entities", [])

                    row = {
                        "frame": p.name,
                        "preview": str(p),
                        "entities": ", ".join([e.get("description", "") for e in entities[:5]]),
                        "similar_count": len(wd.get("visually_similar", [])),
                        "pages_count": len(wd.get("pages_with_matches", [])),
                        "top_urls": ", ".join(urls[:3]),
                        "other_urls": ", ".join(other_urls[:3]),
                        "risk_tags": ", ".join(risk_tags_from_metadata(urls, entities)),
                        "raw_urls": all_urls,
                        "raw_entities": entities,
                    }
                    results.append(row)

                except Exception as e:
                    results.append({
                        "frame": p.name,
                        "preview": str(p),
                        "entities": "",
                        "similar_count": 0,
                        "pages_count": 0,
                        "top_urls": "",
                        "other_urls": "",
                        "risk_tags": f"Error: {e}",
                        "raw_urls": [],
                        "raw_entities": [],
                    })


    # 5) Present findings
    if results:
        st.markdown("---")
        st.subheader("Findings Summary")
        total = len(results)
        with_hits = sum(1 for r in results if (r.get("similar_count", 0) + r.get("pages_count", 0)) > 0)
        flagged = sum(1 for r in results if r.get("risk_tags", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames analyzed", total)
        c2.metric("Frames with hits", with_hits)
        c3.metric("Frames flagged", flagged)

        st.subheader("Findings Table")
        df = pd.DataFrame(results)
        # For display: clickable URLs in a simple way
        def _mk_links(s: str) -> str:
            if not s:
                return ""
            parts = [u.strip() for u in s.split(",") if u.strip()]
            return "<br>".join(
                [f"<a href='{u}' target='_blank'>{u[:80]}{'…' if len(u)>80 else ''}</a>" for u in parts]
            )
            
        disp = df.copy()
        
        for col in ("top_urls", "other_urls"):
            if col in disp.columns:
                disp[col] = disp[col].apply(_mk_links)
    
        
        # Hide raw columns from the view
        drop_cols = [c for c in ("raw_urls", "raw_entities") if c in disp.columns]
        disp_view = disp.drop(columns=drop_cols) if drop_cols else disp

        st.write(disp_view.to_html(escape=False, index=False), unsafe_allow_html=True)


        # Download CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (raw findings)", data=csv_bytes, file_name="findings.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Next Steps / TODOs")
    st.markdown(
        """
        - Add TinEye API as a second source and reconcile results.
        - Integrate stock provider APIs (Adobe Stock first) to fetch license class, releases, keywords.
        - Add OCR (Vision TEXT_DETECTION) & Logo detection; maintain a watchlist to auto-flag.
        - Add a scene timeline UI and per-timecode mapping.
        - Persist job runs in a DB; add a background worker + rate-limiters.
        - CLIP + FAISS to detect reuse across your internal ad corpus.
        """
    )

    # Clean-up button (optional; temp dirs get auto-removed on restart)
    with st.expander("Temporary Files"):
        st.code(str(workdir))
        if st.button("Delete temp workdir"):
            try:
                shutil.rmtree(workdir)
                st.success("Deleted.")
            except Exception as e:
                st.error(f"Failed to delete: {e}")

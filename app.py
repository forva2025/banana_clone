import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# Environment & constants
# -----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
API_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/nano-banana:generateImage"
)


# -----------------------------
# Helpers
# -----------------------------
def validate_inputs(
    mode: str,
    prompt: str,
    image_file: Optional[st.runtime.uploaded_file_manager.UploadedFile],
    num_images: int,
) -> bool:
    """Validate user inputs before API call."""
    if not GOOGLE_API_KEY:
        st.error("Missing GOOGLE_API_KEY. Please add it to your .env and restart the app.")
        return False

    if mode == "Text-to-Image" and not prompt.strip():
        st.error("Please enter a prompt for Text-to-Image mode.")
        return False

    if mode == "Image-to-Image" and image_file is None:
        st.error("Please upload a reference image for Image-to-Image mode.")
        return False

    if num_images < 1 or num_images > 4:
        st.error("Number of images must be between 1 and 4.")
        return False

    return True


def encode_uploaded_image(
    image_file: Optional[st.runtime.uploaded_file_manager.UploadedFile],
) -> Optional[Tuple[str, str]]:
    """Return (base64_str, mime_type) for an uploaded file, or None if not provided."""
    if image_file is None:
        return None
    file_bytes = image_file.read()
    mime_type = image_file.type or "image/png"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return b64, mime_type


def build_request_payload(
    mode: str,
    prompt: str,
    b64_image_and_mime: Optional[Tuple[str, str]],
    size: str,
    num_images: int,
) -> Dict[str, Any]:
    """Construct the request body for Nano Banana generateImage.

    We follow the v1beta models:nano-banana:generateImage format using contents/parts,
    and imageGenerationConfig with numberOfImages and size. The API may evolve; this
    payload aims for compatibility with the current beta.
    """
    parts: List[Dict[str, Any]] = []
    if prompt.strip():
        parts.append({"text": prompt.strip()})

    if mode == "Image-to-Image" and b64_image_and_mime is not None:
        b64, mime_type = b64_image_and_mime
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": b64,
            }
        })

    # Fallback: if no parts (e.g., empty prompt in T2I), ensure at least empty text
    if not parts:
        parts.append({"text": ""})

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "imageGenerationConfig": {
            "numberOfImages": num_images,
            "size": size,
        },
    }

    return payload


def extract_images_from_response(resp_json: Dict[str, Any]) -> List[Tuple[bytes, str]]:
    """Extract images as (bytes, mime_type) from various possible response shapes.

    Returns an empty list if no images found.
    """
    images: List[Tuple[bytes, str]] = []

    # Shape 1: { generatedImages: [ { image: { inlineData: { mimeType, data } } | { imageUrl } } ] }
    if isinstance(resp_json, dict) and "generatedImages" in resp_json:
        for item in resp_json.get("generatedImages", []):
            image_obj = item.get("image") or item
            # inlineData b64
            inline = image_obj.get("inlineData") if isinstance(image_obj, dict) else None
            if inline and inline.get("data"):
                try:
                    images.append((base64.b64decode(inline["data"]), inline.get("mimeType", "image/png")))
                    continue
                except Exception:
                    pass
            # direct URL
            url = image_obj.get("imageUrl") if isinstance(image_obj, dict) else None
            if url:
                try:
                    r = requests.get(url, timeout=30)
                    if r.ok:
                        # Best-effort content-type
                        mime = r.headers.get("Content-Type", "image/png")
                        images.append((r.content, mime))
                        continue
                except Exception:
                    # Ignore URL fetch failure and keep parsing
                    pass

    # Shape 2: candidates -> content -> parts[inline_data]
    if not images:
        for cand in resp_json.get("candidates", []):
            content = cand.get("content", {})
            for part in content.get("parts", []):
                inline = part.get("inline_data")
                if inline and inline.get("data"):
                    try:
                        images.append((base64.b64decode(inline["data"]), inline.get("mime_type", "image/png")))
                    except Exception:
                        pass

    # Shape 3: direct images array with base64 strings
    if not images and isinstance(resp_json.get("images"), list):
        for img in resp_json.get("images", []):
            if isinstance(img, str):
                try:
                    images.append((base64.b64decode(img), "image/png"))
                except Exception:
                    pass
            elif isinstance(img, dict) and img.get("b64"):
                try:
                    images.append((base64.b64decode(img["b64"]), img.get("mime", "image/png")))
                except Exception:
                    pass

    return images


def call_nano_banana(
    payload: Dict[str, Any],
) -> Tuple[Optional[List[Tuple[bytes, str]]], Optional[str]]:
    """Call the Nano Banana API and return (images, error_message)."""
    try:
        url = f"{API_ENDPOINT}?key={GOOGLE_API_KEY}"
        response = requests.post(url, json=payload, timeout=60)
        if not response.ok:
            return None, f"API Error {response.status_code}: {response.text[:500]}"

        data = response.json()
        images = extract_images_from_response(data)
        return images, None
    except Exception as exc:
        return None, f"Request failed: {exc}"


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []  # List[Dict[str, Any]]
    if "current_images" not in st.session_state:
        st.session_state.current_images = []  # List[Tuple[bytes, mime]]
    if "last_params" not in st.session_state:
        st.session_state.last_params = {}


def add_to_history(
    mode: str,
    prompt: str,
    size: str,
    num_images: int,
    images: List[Tuple[bytes, str]],
) -> None:
    st.session_state.history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "prompt": prompt,
            "size": size,
            "num_images": num_images,
            "images": images,  # store bytes in-memory
        },
    )


def render_sidebar_history() -> None:
    st.sidebar.header("History")
    if not st.session_state.history:
        st.sidebar.info("No history yet.")
        return

    for idx, item in enumerate(st.session_state.history):
        label = f"{item['timestamp']} — {item['mode']} — {item['size']}"
        if item.get("prompt"):
            trimmed = item["prompt"][:40] + ("…" if len(item["prompt"]) > 40 else "")
            label += f"\n{trimmed}"
        if st.sidebar.button(label, key=f"hist_{idx}"):
            st.session_state.current_images = item["images"]
            st.session_state.last_params = {
                "mode": item["mode"],
                "prompt": item["prompt"],
                "size": item["size"],
                "num_images": item["num_images"],
            }


def image_bytes_to_download_name(index: int) -> str:
    return f"nano_banana_image_{index+1}.png"


def render_images_grid(images: List[Tuple[bytes, str]]) -> None:
    if not images:
        st.warning("No images to display.")
        return

    # Responsive grid: up to 2 per row for readability
    num_cols = 2 if len(images) > 1 else 1
    rows: List[List[Tuple[bytes, str]]] = []
    for i in range(0, len(images), num_cols):
        rows.append(images[i : i + num_cols])

    for row_idx, row in enumerate(rows):
        cols = st.columns(len(row))
        for col_idx, (img_bytes, mime) in enumerate(row):
            with cols[col_idx]:
                st.image(io.BytesIO(img_bytes), use_column_width=True)
                st.download_button(
                    label="Download",
                    data=img_bytes,
                    file_name=image_bytes_to_download_name(row_idx * num_cols + col_idx),
                    mime=mime or "image/png",
                    key=f"dl_{row_idx}_{col_idx}",
                )


# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Nano Banana Image Generator", page_icon="🍌", layout="wide")
    init_session_state()

    st.title("🍌 Nano Banana Image Generator")
    st.caption("Generate and edit images with Google's Nano Banana (Gemini 2.5 Flash Image)")

    # Sidebar: history
    render_sidebar_history()

    # Mode selector
    mode = st.radio("Mode", ["Text-to-Image", "Image-to-Image"], horizontal=True)

    # Inputs
    default_prompt = st.session_state.last_params.get("prompt", "") if st.session_state.last_params else ""
    prompt = st.text_area("Prompt (optional for Image-to-Image)", value=default_prompt, height=120, placeholder="Describe what you want to generate…")

    size = st.selectbox("Image size", ["512x512", "1024x1024", "2048x2048"], index=1)
    num_images = st.slider("Number of images", 1, 4, st.session_state.last_params.get("num_images", 1) if st.session_state.last_params else 1)

    uploaded_image = None
    if mode == "Image-to-Image":
        uploaded_image = st.file_uploader("Reference image (PNG/JPEG)", type=["png", "jpg", "jpeg"])

    generate = st.button("Generate", type="primary")

    # Show last results if any
    if st.session_state.current_images:
        st.subheader("Current Results")
        render_images_grid(st.session_state.current_images)

    if generate:
        if not validate_inputs(mode, prompt, uploaded_image, num_images):
            return

        with st.spinner("Generating images…"):
            b64_tuple = encode_uploaded_image(uploaded_image) if uploaded_image else None
            payload = build_request_payload(mode, prompt, b64_tuple, size, num_images)
            images, error = call_nano_banana(payload)

        if error:
            st.error(error)
            return

        if not images:
            st.warning("No images were returned by the API.")
            return

        # Update state and display
        st.session_state.current_images = images
        st.session_state.last_params = {
            "mode": mode,
            "prompt": prompt,
            "size": size,
            "num_images": num_images,
        }
        add_to_history(mode, prompt, size, num_images, images)

        st.success("Images generated successfully!")
        st.subheader("Results")
        render_images_grid(images)


if __name__ == "__main__":
    main()


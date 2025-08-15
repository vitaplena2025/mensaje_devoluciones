import io
import re
from datetime import datetime
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st

try:
    import pytesseract
    from pytesseract import Output
except Exception as e:
    pytesseract = None

# -----------------------------
# Helpers
# -----------------------------
NUM_RE = re.compile(r"^\d{4,8}$")  # invoices often 4–8 digits


def _preprocess(img: Image.Image, do_binarize: bool) -> Image.Image:
    """Light preprocessing to help OCR. Keeps original aspect ratio."""
    if not do_binarize:
        return img
    arr = np.array(img.convert("L"))  # grayscale
    # Adaptive threshold for varied lighting
    try:
        import cv2  # type: ignore

        arr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
        )
    except Exception:
        # Fallback simple threshold if OpenCV isn't available
        arr = (arr > 180).astype(np.uint8) * 255
    return Image.fromarray(arr)


def ocr_with_positions(img: Image.Image, lang: str = "spa+eng"):
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract no está disponible. Asegúrate de instalarlo y tener Tesseract OCR en el sistema."
        )
    df = pytesseract.image_to_data(img, lang=lang, output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    # Filtra spans vacíos o basura
    df = df[df["text"].str.len() > 0]
    return df


def _closest_column_x(df, header_words: Tuple[str, ...]) -> Tuple[float, float]:
    """Try to find x-range for a column by locating its header.
    Returns (x_center, tolerance)."""
    mask = df["text"].str.lower().isin(header_words)
    if mask.any():
        h = df[mask].sort_values("conf", ascending=False).iloc[0]
        x_center = h["left"] + h["width"]/2
        tol = max(60.0, h["width"] * 0.8)
        return x_center, tol
    # fallback: pick densest x-position among numeric tokens
    nums = df[df["text"].str.fullmatch(NUM_RE.pattern)]
    if len(nums) == 0:
        return 0.0, 99999.0
    centers = nums["left"] + nums["width"]/2
    # histogram to find the vertical band where most numbers live
    hist, edges = np.histogram(centers, bins=min(40, max(5, int(len(centers)/3))))
    idx = int(np.argmax(hist))
    x_center = float((edges[idx] + edges[idx+1]) / 2)
    tol = max(60.0, float(edges[idx+1] - edges[idx]))
    return x_center, tol


def extract_facturas(img: Image.Image, lang: str = "spa+eng", binarize: bool = True) -> List[str]:
    img2 = _preprocess(img, do_binarize=binarize)
    df = ocr_with_positions(img2, lang=lang)

    # identify the x band for the "Factura" column
    x_center, tol = _closest_column_x(df, header_words=("factura", "facturas"))

    # group by row and collect numbers that fall inside the band
    nums = df[df["text"].str.fullmatch(NUM_RE.pattern)].copy()
    if len(nums) == 0:
        return []

    x_ok = (np.abs((nums["left"] + nums["width"]/2) - x_center) <= tol)
    nums = nums[x_ok]

    # order top-to-bottom, then left-to-right
    nums = nums.sort_values(["top", "left"]).copy()

    # Deduplicate preserving order
    seen = set()
    result: List[str] = []
    for t in nums["text"].tolist():
        # normalize (remove leading zeros just in case OCR gives them)
        t_norm = str(int(t)) if t.isdigit() else t
        if t_norm not in seen:
            seen.add(t_norm)
            result.append(t_norm)
    return result


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Devoluciones • Facturas OCR", page_icon="📄")

st.title("📄 Devoluciones – Extractor de Facturas (OCR)")
st.write(
    "Sube una captura o foto del **Registro de Devoluciones**. El sistema leerá la columna *Factura*, "
    "removerá duplicados y creará un mensaje listo para WhatsApp."
)

with st.sidebar:
    st.header("Ajustes OCR")
    lang = st.selectbox("Idioma OCR", ["spa+eng", "spa", "eng"], index=0)
    binarize = st.checkbox("Mejorar contraste (recomendado)", value=True)

uploaded = st.file_uploader("Subir imagen (PNG/JPG)", type=["png", "jpg", "jpeg"]) 

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Vista previa", use_column_width=True)

    with st.spinner("Leyendo columna 'Factura'…"):
        try:
            facturas = extract_facturas(image, lang=lang, binarize=binarize)
        except Exception as e:
            st.error(str(e))
            facturas = []

    hoy = datetime.now().strftime("%d/%m/%Y")
    if facturas:
        mensaje = f"Devoluciones ({hoy}): " + ", ".join(facturas)
        st.success("¡Listo! Mensaje generado.")
        st.code(mensaje, language="text")
        st.download_button(
            label="Descargar mensaje .txt",
            data=mensaje.encode("utf-8"),
            file_name=f"devoluciones_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )

        with st.expander("Ver números detectados"):
            st.write(f"Total: {len(facturas)} (únicos)")
            st.write(facturas)
    else:
        st.warning(
            "No se identificaron números de factura. Prueba con *Mejorar contraste*, cambia el idioma o sube una imagen más nítida."
        )

st.markdown(
    "— *Sugerencia:* Si el encabezado **Factura** no es reconocido, el sistema intenta ubicar automáticamente la banda vertical con mayor densidad de números."
)

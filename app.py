import streamlit as st
import easyocr
import pytesseract
import numpy as np
import cv2
from PIL import Image
import re
from sympy import symbols, Eq, solve, sympify, pi, sqrt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ---------------------------
# 1. Image Preprocessing
# ---------------------------
def preprocess_image(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    base_height = 600
    h, w = img.shape[:2]
    if h > base_height:
        scale = base_height / float(h)
        img = cv2.resize(img, (int(w * scale), base_height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 11, 90, 90)
    kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel_sharp)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    bin_for_ocr = 255 - th
    return img, bin_for_ocr

# ---------------------------
# 2. OCR with EasyOCR & Tesseract
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def easyocr_extract(img):
    reader = get_easyocr_reader()
    allowed_chars = "0123456789xX+-*/=()^√π"
    results = reader.readtext(img, detail=1, allowlist=allowed_chars, paragraph=True)
    text = " ".join([res[1] for res in results])
    return text

def pytesseract_extract(img):
    config = "--psm 7 -c tessedit_char_whitelist=0123456789xX+-*/=()^√π"
    text = pytesseract.image_to_string(img, config=config)
    return text

# ---------------------------
# 3. Fuzzy Correction & Cleaning
# ---------------------------
def clean_ocr_text(text):
    replacements = {
        "X": "x", "×": "*", "−": "-", "—": "-", ":": "=",
        "l": "1", "S": "5", "O": "0", "B": "8", " ": ""
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^0-9xX+\-*/=().^√π]", "", text)
    text = re.sub(r'([+\-*/=^])\1+', r'\1', text) # Replace repeated operators
    return text

def correct_equation(equation_str):
    if "=" not in equation_str and equation_str.count('-') >= 1:
        last_hyphen_index = equation_str.rfind('-')
        equation_str = equation_str[:last_hyphen_index] + '=' + equation_str[last_hyphen_index + 1:]
        st.info(f"Corrected equation assumption: {equation_str}")
    return equation_str

# ---------------------------
# 4. Solve the Equation
# ---------------------------
def solve_equation(equation_str):
    if not equation_str:
        return "Empty equation."
    try:
        x = symbols("x")
        transformations = (standard_transformations + (implicit_multiplication_application,))
        processed_str = equation_str.replace("√", "sqrt").replace("π", "pi")
        processed_str = re.sub(r'(\d)([xX(])', r'\1*\2', processed_str)
        processed_str = re.sub(r'([xX)])(\d)', r'\1*\2', processed_str)
        if "=" in processed_str:
            parts = processed_str.split("=", 1)
            left = parts[0] if parts[0] else '0'
            right = parts[1] if len(parts) > 1 and parts[1] else '0'
            eq = Eq(parse_expr(left, transformations=transformations), parse_expr(right, transformations=transformations))
        else:
            eq = Eq(parse_expr(processed_str, transformations=transformations), 0)
        solution = solve(eq, x)
        return eq, solution
    except Exception as e:
        return None, f"Error: Could not solve the equation. ({e})"

# ---------------------------
# 5. Streamlit App (Modern UI)
# ---------------------------
st.set_page_config(page_title="Handwritten Math Solver", page_icon="🧮", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🧮 Math Solver")
    st.markdown("""
    **Instructions:**
    1. Write your equation clearly on white paper.
    2. Take a well-lit photo or scan.
    3. Upload the image below.
    4. The app will OCR and solve for $x$.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) and [EasyOCR](https://github.com/JaidedAI/EasyOCR).")

st.markdown("<h1 style='color:#4F8BF9;'>📝 Handwritten Math Equation Solver</h1>", unsafe_allow_html=True)
st.caption("Upload a clear image of a handwritten equation (e.g., <code>2x+3=7</code>, <code>4(x-1)=10</code>). The app will OCR and solve for <b>x</b>.")

uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(pil_img, caption="Original", use_column_width=True)
    original, processed = preprocess_image(pil_img)
    with col2:
        st.subheader("🖼️ Preprocessing Preview")
        st.image(processed, caption="Processed for OCR", use_column_width=True)

    st.markdown("---")
    st.subheader("🔍 OCR Results & Equation")
    with st.spinner('Performing OCR...'):
        raw_easy = easyocr_extract(processed)
        try:
            raw_tess = pytesseract_extract(processed)
        except pytesseract.TesseractNotFoundError:
            raw_tess = ""
            st.warning("Tesseract is not installed or not in your PATH. The app will rely only on EasyOCR.")

        cleaned_easy = clean_ocr_text(raw_easy)
        cleaned_tess = clean_ocr_text(raw_tess)
        final_raw = raw_easy if len(cleaned_easy) >= len(cleaned_tess) else raw_tess

    with st.expander("🔤 Show OCR Outputs"):
        st.write(f"**EasyOCR Output:** `{raw_easy}`")
        if raw_tess:
            st.write(f"**Tesseract Output:** `{raw_tess}`")

    cleaned = clean_ocr_text(final_raw)
    corrected = correct_equation(cleaned)
    st.info(f"**Equation Used:** `{corrected}`")

    st.markdown("---")
    st.subheader("🧮 Solution")
    eq, solution = solve_equation(corrected)
    if eq is not None:
        st.latex(f"{eq}")
        if isinstance(solution, list) and len(solution) > 0:
            sol_latex = ", ".join([f"x = {s.evalf()}" for s in solution])
            st.success(f"**Solution:** {sol_latex}")
        elif isinstance(solution, list) and len(solution) == 0:
            st.warning("No solution found for $x$.")
        else:
            st.info(f"Result: {solution}")
    else:
        st.error(solution)

    st.markdown("---")
    if st.button("🔄 Reset"):
        st.experimental_rerun()
else:
    st.info("Upload an image to begin.")

st.markdown("""
---
<small>
<b>Tips for best results:</b>
- Write clearly with a dark pen/marker.
- Use plain white paper and good lighting.
- Crop the image to the equation only.
- Supported: Linear equations in one variable (e.g., <code>2x+3=7</code>).
</small>
""", unsafe_allow_html=True)
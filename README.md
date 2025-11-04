# handwritten-math-equation-solver
A Python project that solves handwritten linear algebraic equations using image processing and OCR.


This project is a **Streamlit web app** that reads handwritten **linear algebraic equations** from an image, extracts the text using **EasyOCR + Tesseract**, and solves for **x** using **SymPy**.  
Just upload a clear image of your equation, and get the solution in seconds 

## 💡 Features
- 🧠 Reads **handwritten equations** using EasyOCR and Tesseract  
- 🔍 Smart **image preprocessing** with OpenCV for better OCR accuracy  
- ➗ Solves **linear algebraic equations** (like `2x + 3 = 7`)  
- 📊 Clean, modern **Streamlit UI**  
- ⚙️ Real-time equation correction and parsing

- ## 🛠️ Installation & Usage

- ## 🧰 Tech Stack
| Purpose | Tool / Library |
|----------|----------------|
| Web App | [Streamlit](https://streamlit.io/) |
| OCR Engine | [EasyOCR](https://github.com/JaidedAI/EasyOCR) + [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) |
| Image Processing | OpenCV |
| Math Solver | SymPy |
| Language | Python 🐍 |

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/handwritten-math-equation-solver.git
cd handwritten-math-equation-solver


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Run the app

streamlit run app.py






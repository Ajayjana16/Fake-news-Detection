"""
Fake News Detection Flask Backend API

This backend provides:
- GET  /api/health            -> System health check and model loading status
- POST /api/predict           -> Model prediction using pre-trained Colab artifacts (supports text, URL, and document file)
- POST /api/extract-file      -> Extracts headline and text from PDF, DOCX, and TXT files
- POST /api/extract-url       -> Fetches and extracts headline and text from a news website URL
- GET  /api/metrics           -> Model comparison evaluation metrics loaded from CSV
- Static file serving for the frontend web application
"""

import os
import io
import csv
import re
import logging
import warnings
import urllib.parse
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import docx

# Suppress sklearn unpickling version warnings for clean server logs
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from preprocessing import preprocess_text, combine_news_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")


def resolve_file_path(relative_path: str) -> str:
    """
    Locates a file either under backend/ or project root.
    """
    candidates = [
        os.path.join(BASE_DIR, relative_path),
        os.path.join(ROOT_DIR, relative_path),
        os.path.join(os.getcwd(), relative_path),
        os.path.join(os.getcwd(), "backend", relative_path),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return os.path.abspath(os.path.join(BASE_DIR, relative_path))


# ==============================================================================
# VERIFIED LABEL MAPPING
# ==============================================================================
# In the trained models (trained on the ISOT Fake/Real News dataset):
# Class 0: Negative coefficients in LR/SVM (words like 'via', 'video', 'breaking', 'wire') -> FAKE NEWS
# Class 1: Positive coefficients in LR/SVM (words like 'reuters', 'said', 'washington')    -> REAL NEWS
LABEL_MAPPING = {
    0: "FAKE NEWS",
    1: "REAL NEWS"
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest"
}

# Recognized accredited news organizations and wire agencies
VERIFIED_NEWS_DOMAINS = {
    "timesofindia.indiatimes.com": "The Times of India",
    "indiatimes.com": "Times of India",
    "thehindu.com": "The Hindu",
    "indianexpress.com": "The Indian Express",
    "hindustantimes.com": "Hindustan Times",
    "ndtv.com": "NDTV",
    "reuters.com": "Reuters",
    "bbc.com": "BBC News",
    "bbc.co.uk": "BBC News",
    "apnews.com": "Associated Press",
    "aljazeera.com": "Al Jazeera",
    "cnn.com": "CNN",
    "bloomberg.com": "Bloomberg",
    "theguardian.com": "The Guardian",
    "economictimes.indiatimes.com": "The Economic Times",
    "tribuneindia.com": "The Tribune",
    "deccanherald.com": "Deccan Herald",
    "indiatoday.in": "India Today",
    "theprint.in": "ThePrint",
    "thequint.com": "The Quint",
    "news18.com": "News18",
    "firstpost.com": "Firstpost",
    "livemint.com": "Mint",
    "business-standard.com": "Business Standard",
    "ptinews.com": "Press Trust of India (PTI)",
    "aninews.in": "Asian News International (ANI)",
    "wikipedia.org": "Wikipedia"
}

DATELINE_REGEX = re.compile(
    r"\b(NEW DELHI|WASHINGTON|LONDON|BEIJING|MOSCOW|PARIS|BERLIN|TOKYO|ISLAMABAD|MUMBAI|KOLKATA|CHENNAI|BENGALURU|HYDERABAD|GENEVA|NEW YORK|KYIV)\s*[:\-\—\(]|"
    r"\b(Reuters|AP|AFP|PTI|ANI|IANS|Bloomberg|TNN)\b",
    re.I
)

def check_source_verification(domain: str, raw_text: str):
    """
    Checks if an article is attributed to an accredited news publisher or legitimate wire dateline.
    """
    domain = (domain or "").lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    for d, name in VERIFIED_NEWS_DOMAINS.items():
        if domain == d or domain.endswith("." + d):
            return True, f"Verified Publisher ({name})"
    if DATELINE_REGEX.search((raw_text or "")[:400]):
        return True, "Verified Press/Wire Dateline"
    return False, None

# ==============================================================================
# ARTIFACT INITIALIZATION
# ==============================================================================
models: Dict[str, Any] = {}
tfidf_vectorizer = None
artifacts_loaded = False
load_error_message: Optional[str] = None


def load_artifacts():
    global models, tfidf_vectorizer, artifacts_loaded, load_error_message
    try:
        lr_path = resolve_file_path(os.path.join("models", "logistic_regression_model.joblib"))
        svm_path = resolve_file_path(os.path.join("models", "svm_model.joblib"))
        rf_path = resolve_file_path(os.path.join("models", "random_forest_model.joblib"))
        tfidf_path = resolve_file_path(os.path.join("models", "tfidf_vectorizer.joblib"))

        logger.info(f"Loading TF-IDF vectorizer from {tfidf_path}")
        tfidf_vectorizer = joblib.load(tfidf_path)

        logger.info(f"Loading Logistic Regression from {lr_path}")
        models["logistic_regression"] = joblib.load(lr_path)

        logger.info(f"Loading SVM from {svm_path}")
        models["svm"] = joblib.load(svm_path)

        logger.info(f"Loading Random Forest from {rf_path}")
        models["random_forest"] = joblib.load(rf_path)

        artifacts_loaded = True
        logger.info("All trained artifacts successfully loaded into memory.")
    except Exception as exc:
        artifacts_loaded = False
        load_error_message = str(exc)
        logger.error(f"Error loading trained artifacts: {exc}")


load_artifacts()

# Flask App setup
app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
CORS(app)


# ==============================================================================
# DOCUMENT & URL EXTRACTION UTILITIES
# ==============================================================================

def extract_text_from_file_data(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text content and derived headline from PDF, DOCX, or TXT documents.
    """
    if not filename or '.' not in filename:
        raise ValueError("Invalid file. Please provide a file with a valid extension (.pdf, .docx, .txt).")

    ext = filename.lower().rsplit('.', 1)[-1]

    if ext == 'pdf':
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for page in reader.pages:
                t = page.extract_text() or ''
                if t.strip():
                    pages_text.append(t.strip())
            full_text = "\n\n".join(pages_text).strip()
        except Exception as e:
            raise ValueError(f"Failed to parse PDF document: {str(e)}")

        if not full_text:
            raise ValueError("The uploaded PDF file contains no selectable text (it may be a scanned image or protected document).")

        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        title = lines[0] if lines else os.path.splitext(filename)[0]
        article_text = "\n\n".join(lines[1:]) if len(lines) > 1 else full_text

        return {
            "title": title[:200],
            "text": article_text,
            "filename": filename,
            "file_type": "PDF",
            "page_count": len(reader.pages)
        }

    elif ext == 'docx':
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        except Exception as e:
            raise ValueError(f"Failed to parse Word document: {str(e)}")

        if not paragraphs:
            raise ValueError("The uploaded Word document contains no readable text.")

        title = paragraphs[0] if len(paragraphs) > 1 else os.path.splitext(filename)[0]
        article_text = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else paragraphs[0]

        return {
            "title": title[:200],
            "text": article_text,
            "filename": filename,
            "file_type": "DOCX",
            "paragraph_count": len(paragraphs)
        }

    elif ext in ['txt', 'text']:
        try:
            full_text = file_bytes.decode('utf-8').strip()
        except UnicodeDecodeError:
            full_text = file_bytes.decode('latin-1', errors='ignore').strip()

        if not full_text:
            raise ValueError("The uploaded text file is empty.")

        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        title = lines[0] if lines else os.path.splitext(filename)[0]
        article_text = "\n\n".join(lines[1:]) if len(lines) > 1 else full_text

        return {
            "title": title[:200],
            "text": article_text,
            "filename": filename,
            "file_type": "TXT",
            "line_count": len(lines)
        }

    else:
        raise ValueError(f"Unsupported file format (.{ext}). Supported formats: .pdf, .docx, .txt")


def extract_text_from_url_data(url: str) -> Dict[str, Any]:
    """
    Fetches a webpage from a URL and extracts its article headline and body text.
    Supports Schema.org JSON-LD NewsArticle structured data (standard in major outlets
    like Times of India, BBC, Reuters, CNN, NDTV, The Hindu) and provides robust
    DOM parsing with boilerplate filtering as fallback.
    """
    url = (url or "").strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc:
        raise ValueError("Invalid URL format. Please enter a valid website link (e.g. https://www.reuters.com/...)")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("The request to the news website timed out. The server took too long to respond.")
    except requests.exceptions.SSLError:
        raise ValueError("SSL certificate verification failed for the requested URL.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Could not connect to the specified website. Please check the URL and ensure the domain exists.")
    except requests.exceptions.HTTPError:
        status_code = response.status_code if 'response' in locals() and response else 'error'
        raise ValueError(f"Web server returned HTTP error {status_code} when accessing the article.")
    except Exception as e:
        raise ValueError(f"Failed to access the website link: {str(e)}")

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ValueError(f"URL did not return an HTML webpage (Content-Type: {content_type}).")

    soup = BeautifulSoup(response.content, "html.parser")

    title = ""
    article_text = ""

    # --------------------------------------------------------------------------
    # 1. Primary Strategy: Schema.org JSON-LD (NewsArticle / Article / Report)
    # Extracts clean author-intended body without ads, navigation bars, or edition menus
    # --------------------------------------------------------------------------
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            raw = (s.string or s.text or "").strip()
            if not raw:
                continue
            data = json.loads(raw)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                graph = item.get("@graph", [])
                candidates = [item] + (graph if isinstance(graph, list) else ([graph] if isinstance(graph, dict) else []))
                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    if not title and cand.get("headline"):
                        cand_hl = str(cand.get("headline")).strip()
                        if len(cand_hl) > 5:
                            title = cand_hl
                    body = cand.get("articleBody") or cand.get("article_body") or cand.get("text")
                    if body and len(str(body).strip()) > 80:
                        article_text = str(body).strip()
                        break
                if article_text:
                    break
            if article_text:
                break
        except Exception:
            continue

    # Headline fallback if not present in JSON-LD
    if not title:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("meta", attrs={"name": "twitter:title"}):
            title = soup.find("meta", attrs={"name": "twitter:title"}).get("content", "").strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

    # Clean brand suffix from title (e.g. " - The Times of India", " | Reuters")
    title = re.sub(
        r"\s*[-|–—]\s*(The Times of India|Times of India|BBC News|Reuters|CNN|The Hindu|NDTV|Hindustan Times|Indian Express|Wikipedia).*$",
        "",
        title,
        flags=re.I
    ).strip()

    # --------------------------------------------------------------------------
    # 2. Secondary Strategy: DOM Parsing (for sites without JSON-LD or brief summaries)
    # --------------------------------------------------------------------------
    if not article_text or len(article_text) < 100:
        soup_clean = BeautifulSoup(response.content, "html.parser")

        # Strip non-content interactive and decorative elements
        for tag in list(soup_clean(["script", "style", "nav", "footer", "header", "form", "svg", "noscript", "aside", "iframe", "button", "select"])):
            try:
                tag.decompose()
            except Exception:
                pass

        # Remove header/navigation wrappers while preserving news containers and root elements
        for junk in list(soup_clean.find_all(["div", "section", "ul", "ol", "span", "aside", "p"], class_=re.compile(r"\b(nav|navigation|edition-picker|site-header|site-footer|share-bar|social-share|comment-box|sidebar|ad-container)\b", re.I))):
            if not getattr(junk, "attrs", None) or junk.name in ["html", "body", "main", "article"]:
                continue
            cls_val = junk.attrs.get("class", "")
            cls_str = " ".join(cls_val) if isinstance(cls_val, list) else str(cls_val)
            if not any(good in cls_str.lower() for good in ["story", "article-body", "articlebody", "js_tbl_article", "fewcent", "mw-parser-output"]):
                try:
                    junk.decompose()
                except Exception:
                    pass

        # Specialized news article containers
        container_candidates = [
            soup_clean.find(attrs={"data-articlebody": True}),
            soup_clean.find(attrs={"itemprop": "articleBody"}),
            soup_clean.find("div", class_=re.compile(r"js_tbl_article|fewcent", re.I)),
            soup_clean.find("div", class_=re.compile(r"article[-_]?body|story[-_]?body|story[-_]?content|entry[-_]?content|post[-_]?content|mw-parser-output", re.I)),
            soup_clean.find("article"),
            soup_clean.find("main"),
            soup_clean.body,
        ]

        best_container = None
        for c in container_candidates:
            if c and len(c.get_text().strip()) > 80:
                best_container = c
                break

        if not best_container:
            best_container = soup_clean

        # Extract paragraphs
        paragraphs = []
        for p in best_container.find_all("p"):
            txt = p.get_text().strip()
            if len(txt) > 25:
                low = txt.lower()
                if not any(bad in low for bad in ["cookie policy", "all rights reserved", "terms of service", "subscribe now", "sign up", "javascript required"]):
                    paragraphs.append(txt)

        if paragraphs:
            article_text = "\n\n".join(paragraphs)
        else:
            # Fallback for sites where body is inside nested divs without <p> tags
            lines = [l.strip() for l in best_container.get_text(separator="\n").splitlines() if len(l.strip()) > 35]
            article_text = "\n\n".join(lines)

    # --------------------------------------------------------------------------
    # 3. Tertiary Fallback: OpenGraph / Meta Description
    # --------------------------------------------------------------------------
    if not article_text.strip():
        meta_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content") and len(meta_desc["content"].strip()) > 30:
            article_text = meta_desc["content"].strip()

    if not article_text.strip():
        raise ValueError("Could not extract readable article text from the provided webpage.")

    clean_domain = parsed.netloc.lower()
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]

    return {
        "title": title[:200],
        "text": article_text,
        "url": url,
        "domain": clean_domain
    }


# ==============================================================================
# API ROUTES
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning system status and model loading readiness.
    """
    if not artifacts_loaded:
        return jsonify({
            "status": "error",
            "message": f"Artifacts not loaded: {load_error_message}",
            "models_loaded": list(models.keys())
        }), 503

    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys()),
        "label_mapping": LABEL_MAPPING,
        "features": {
            "text_input": True,
            "document_upload": True,
            "url_extraction": True,
            "supported_document_types": ["pdf", "docx", "txt"]
        }
    }), 200


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Reads and returns the exact performance evaluation results from model_comparison_results.csv.
    Does not invent or calculate artificial numbers.
    """
    csv_path = resolve_file_path(os.path.join("results", "model_comparison_results.csv"))

    if not os.path.exists(csv_path):
        return jsonify({
            "status": "error",
            "message": "Model comparison results unavailable. CSV file not found."
        }), 404

    try:
        results = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    "model": row.get("Model", "").strip(),
                    "accuracy": float(row.get("Accuracy", 0.0)),
                    "precision": float(row.get("Precision", 0.0)),
                    "recall": float(row.get("Recall", 0.0)),
                    "f1_score": float(row.get("F1-Score", 0.0))
                })

        return jsonify({
            "status": "ok",
            "source": "model_comparison_results.csv",
            "data": results
        }), 200
    except Exception as exc:
        logger.error(f"Error reading comparison metrics: {exc}")
        return jsonify({
            "status": "error",
            "message": "Model comparison results unavailable."
        }), 500


@app.route("/api/extract-file", methods=["POST"])
def extract_file():
    """
    Extracts title and text content from an uploaded document (PDF, DOCX, TXT).
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Please attach a PDF, Word DOCX, or TXT file."}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        file_bytes = file.read()
        extracted = extract_text_from_file_data(file.filename, file_bytes)
        return jsonify({
            "status": "ok",
            "data": extracted
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        logger.error(f"File extraction error: {exc}")
        return jsonify({"error": "Failed to extract text from the uploaded document."}), 500


@app.route("/api/extract-url", methods=["POST"])
def extract_url():
    """
    Fetches and extracts headline and text content from a news article URL.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    data = request.get_json()
    url = data.get("url", "").strip() if data else ""

    if not url:
        return jsonify({"error": "Please provide a valid website URL."}), 400

    try:
        extracted = extract_text_from_url_data(url)
        return jsonify({
            "status": "ok",
            "data": extracted
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        logger.error(f"URL extraction error: {exc}")
        return jsonify({"error": f"Failed to extract content from the URL: {str(exc)}"}), 500


@app.route("/api/predict", methods=["POST"])
def predict_news():
    """
    Predicts whether a news item is Fake News or Real News using selected trained model.
    
    Supports:
    1. Direct JSON: {"title": "...", "text": "...", "model": "..."}
    2. URL mode: {"url": "https://...", "model": "..."}
    3. File upload via multipart/form-data: file + model
    """
    if not artifacts_loaded:
        return jsonify({
            "error": "Trained models are not loaded. Please verify model artifacts."
        }), 503

    title = ""
    text = ""
    model_key = "logistic_regression"
    source_type = "text"
    source_info = ""

    # Case A: Multipart form upload
    if request.content_type and "multipart/form-data" in request.content_type:
        model_key = request.form.get("model", "logistic_regression")
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            try:
                extracted = extract_text_from_file_data(file.filename, file.read())
                title = extracted["title"]
                text = extracted["text"]
                source_type = "file"
                source_info = f"{extracted['filename']} ({extracted['file_type']})"
            except ValueError as ve:
                return jsonify({"error": str(ve)}), 400
        else:
            title = request.form.get("title", "")
            text = request.form.get("text", "")

    # Case B: JSON payload
    elif request.is_json:
        data = request.get_json() or {}
        model_key = data.get("model", "logistic_regression")

        # Sub-case: URL provided in JSON
        if "url" in data and data["url"].strip():
            url = data["url"].strip()
            try:
                extracted = extract_text_from_url_data(url)
                title = extracted["title"]
                text = extracted["text"]
                source_type = "url"
                source_info = extracted["domain"]
            except ValueError as ve:
                return jsonify({"error": str(ve)}), 400
        else:
            title = data.get("title", "")
            text = data.get("text", "")
            source_type = "text"
            source_info = "Direct text input"

    else:
        return jsonify({"error": "Unsupported request format. Send JSON or multipart/form-data."}), 400

    # Validate model selection
    if model_key not in models:
        valid_options = ", ".join(MODEL_DISPLAY_NAMES.keys())
        return jsonify({
            "error": f"Invalid model selection: '{model_key}'. Supported models: {valid_options}"
        }), 400

    # Combine title and text exactly matching training convention
    raw_content = combine_news_inputs(title=title, text=text)

    if not raw_content.strip():
        return jsonify({
            "error": "Please provide a news title, article text, document, or website link to analyze."
        }), 400

    # Apply identical 10-step preprocessing
    cleaned_content = preprocess_text(raw_content)

    if not cleaned_content.strip():
        return jsonify({
            "error": "The input text contains no usable alphabetic words after removing stopwords and special characters."
        }), 400

    try:
        # Transform using saved TF-IDF vectorizer (transform ONLY, no fit)
        tfidf_features = tfidf_vectorizer.transform([cleaned_content])

        # Infer using chosen trained model
        selected_model = models[model_key]
        raw_prediction = int(selected_model.predict(tfidf_features)[0])

        # Dual-layer verification: Check if article originates from a verified publisher or accredited press dateline
        is_verified_source, verification_detail = check_source_verification(
            domain=source_info if source_type == "url" else "",
            raw_text=raw_content
        )

        # If verified reputable publisher or accredited wire dateline, calibrate classification to REAL NEWS
        if is_verified_source:
            raw_prediction = 1

        # Map to verified human-readable label
        mapped_label = LABEL_MAPPING.get(raw_prediction, "UNKNOWN")
        display_model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        return jsonify({
            "prediction": mapped_label,
            "raw_class": raw_prediction,
            "model": display_model_name,
            "model_key": model_key,
            "token_count": len(cleaned_content.split()),
            "title": title,
            "source_type": source_type,
            "source_info": source_info,
            "verified_source": is_verified_source,
            "verification_detail": verification_detail or ""
        }), 200

    except Exception as exc:
        logger.error(f"Inference error with model {model_key}: {exc}")
        return jsonify({
            "error": "An internal error occurred during prediction analysis."
        }), 500


# ==============================================================================
# STATIC FRONTEND ROUTES
# ==============================================================================

@app.route("/", methods=["GET"])
def serve_index():
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({
        "message": "Fake News Detection API is active. Frontend static files not found in frontend directory."
    }), 200


@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"error": "Resource not found"}), 404


if __name__ == "__main__":
    logger.info("Starting Fake News Detection Flask API on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

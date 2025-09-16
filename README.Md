Stock Footage Identifier — Streamlit MVP
---------------------------------------
A tiny Streamlit prototype that:
• uploads an ad video
• extracts representative frames via scene detection (PySceneDetect) or 1 fps fallback (ffmpeg)
• de-duplicates frames using imagehash (pHash)
• sends up to N sample frames to Google Cloud Vision Web Detection
• shows a findings table with basic risk tags + an export CSV


Setup (local):
1) Python 3.10+
2) Install ffmpeg (macOS: brew install ffmpeg, Ubuntu: apt-get install ffmpeg)
3) pip install -r requirements.txt (see the list below)
4) Create a Google Cloud project → enable Vision API → download a service account JSON key
and set the env var BEFORE running Streamlit:
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/your-sa.json"
5) Run the app:
streamlit run app.py

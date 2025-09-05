## 🍌 Nano Banana Image Generator

Generate and edit images using Google's Nano Banana (Gemini 2.5 Flash Image) model in a single-file Streamlit app. Supports Text-to-Image and Image-to-Image (variation/edit) modes, with history and downloads.

### Features
- Text-to-Image and Image-to-Image modes
- Size options: 512x512, 1024x1024, 2048x2048
- Generate 1–4 images per request
- Responsive image grid with download buttons
- Sidebar history of past generations; click to reload results

### Requirements
- Python 3.9+
- A Google API key with access to Nano Banana

### Setup
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add a `.env` file with your API key (already created as a placeholder):

```bash
GOOGLE_API_KEY=your_api_key_here
```

### Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Notes
- The app calls:
  `https://generativelanguage.googleapis.com/v1beta/models/nano-banana:generateImage?key=YOUR_KEY`
- The response parser handles base64 and URL image formats.
- Your prompt+image results are kept in memory via Streamlit `st.session_state`.


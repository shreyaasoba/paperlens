# PaperLens

An AI research companion, built for the Girl Develop It **Build Beyond 2026** hackathon.
Upload any research paper and get:

1. a **plain summary** at your level (Beginner, Developer or Researcher)
2. a **jargon buster**: every hard term with a simple meaning and an everyday analogy
3. a **method flow**: the AI lists the steps, our code draws the timeline
4. **checked claims**: every claim comes with a quote, and our code checks the quote really is in the paper

## How it works

```
You upload a PDF ─► READ  read_upload()   PDF pages become plain text (pypdf)
                 ─► ASK   analyze()       prompt + paper text + the shape we want ─► Gemini
                 ─► SHAPE PaperBreakdown  check the JSON answer has every field (pydantic)
                 ─► CHECK quote_found()   is each quote really in the paper?
                 ─► SHOW  st.tabs         four tabs in the browser (Streamlit)
If the AI call fails, the app shows a prepared example instead.
```

## What's in the `app` folder

| File | What it is |
|---|---|
| `app.py` | The whole app, in one file |
| `requirements.txt` | Packages to install |
| `sample_paper.txt` | A short sample (a paraphrase of *Attention Is All You Need*, 2017) |
| `demo.html` | A no-setup preview: open it in any browser. Prepared examples only, no AI calls |
| `.env.example` | The two settings you need. Your real key goes in your terminal, never in a file |
| `.gitignore` | Stops secrets and installed packages being uploaded to GitHub |
| `.streamlit/config.toml` | App settings |

## Run it on your laptop

You need Python 3.10 or newer.

**1. Get a free Gemini API key**
Go to https://aistudio.google.com, sign in with Google, click **Get API key**, then **Create API key**. Keep it private.

**2. Install (one time)**

Mac / Linux:
```
cd paperlens_gdi_2026/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows (Command Prompt):
```
cd paperlens_gdi_2026\app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run**

Mac / Linux:
```
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-2.5-flash"
streamlit run app.py
```
Windows:
```
set GEMINI_API_KEY=your-key
set GEMINI_MODEL=gemini-2.5-flash
streamlit run app.py
```
Your browser opens at http://localhost:8501. Upload a paper, choose a level and click **Explain this paper**.

To see which models your key can use:
```
python -c "from google import genai; c=genai.Client(); print([m.name for m in c.models.list() if 'flash' in m.name])"
```
Pick a plain "flash" name (not lite, tts, image, audio or preview) and drop the `models/` part.

No key yet? The app still runs: leave **Live AI** off to explore the prepared example.

## Weekend ideas
- **Beginner:** add a "real-world example" field, or explain papers in another language.
- **Intermediate:** show the page each quote came from; add a quiz.
- **Advanced:** upload a new paper and the classic it builds on, and explain what changed.

## Good habits
- The free Gemini tier has rate limits, and Google may use free-tier inputs to improve its products. Use public papers only.
- Never upload a paper you're peer-reviewing, or anything confidential.
- Valid JSON proves the shape is right, not the facts. The quote check is a first test, not full verification.

Made by Shreya Asoba · shreyaasoba.com

# Jayanta AI — West Bengal Institute & Course Finder

## Included
- Groq API remains the model provider. Default model: `openai/gpt-oss-120b`.
- Bengali, English and Hindi language selection is sent to the backend.
- Lightweight DuckDuckGo HTML search with source URLs and snippets.
- Google Maps search link for location-related queries.
- Safer frontend rendering (user/model text is not injected as HTML).
- Production debug disabled; no hard-coded API key or Flask secret.
- `/health` endpoint for basic health checks.

## Before deployment
1. Keep a backup of the currently deployed source and note the current Git commit.
2. Check the files before replacing your GitHub branch.
3. This project does not contain a comprehensive verified institute database. Search results are discovery hints, not proof. Always verify fees, admissions and recognition with official sources.
4. DuckDuckGo HTML search is best-effort and may be rate-limited or blocked. For production-grade coverage, replace it with a supported search API and/or curated official institute dataset.
5. Route support currently provides Google Maps search. Turn-by-turn public transport instructions should be checked in Maps; this app does not calculate live transit timetables.

## Render settings
Service: `https://gemini-course-bot.onrender.com`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --workers 2 --threads 4 --timeout 90`
- Environment Variables:
  - `GROQ_API_KEY` = your existing Groq API key (set privately in Render; do not paste it into code/chat)
  - `GROQ_MODEL` = `openai/gpt-oss-120b` (optional; keep current model unless you intentionally change it)
  - `SECRET_KEY` = a long random secret generated for this service (recommended; do not commit it to GitHub)

Do not enable Flask debug mode in production. After deployment, test:
- `https://gemini-course-bot.onrender.com/health`
- Bengali: `হুগলিতে Montessori training institute খুঁজুন`
- English: `Find vocational training institutes in Howrah and give official sources`
- Hindi: `पश्चिम बंगाल में नर्सिंग कोर्स के लिए संस्थान खोजें`
- Route: `Howrah station से इस संस्थान तक कैसे जाएँ?` (the bot should ask for a destination if unclear)

## Rollback
If the new deployment fails, use GitHub's commit history to revert to the previously recorded working commit, or restore the backup copy. Do not delete the old working source until the new deployment passes testing.

# Jayanta AI — West Bengal Institute & Course Finder

## Included
- Groq API remains the model provider. Default model: `openai/gpt-oss-120b`.
- Bengali, English and Hindi language selection is sent to the backend.
- Lightweight DuckDuckGo HTML search with source URLs and snippets.
- Place lookup: Google Places API (New) when a working key is configured, otherwise OpenStreetMap Nominatim. If neither finds the institute, a precise Google Maps search link (institute + area) is offered instead.
- The user's question is first split by the LLM into institute / locality / clean English + original-script queries, so follow-ups like "what is its phone number?" and Bengali spellings of places also work.
- Safer frontend rendering (user/model text is not injected as HTML).
- Production debug disabled; no hard-coded API key or Flask secret.
- Chat history is kept in the browser and sent with each request (no cookie size limit, works across gunicorn workers). A "New chat" button clears it.
- Bot replies are rendered as sanitized Markdown (marked + DOMPurify from cdnjs); falls back to plain text if the libraries cannot load.
- `/health` endpoint for basic health checks.

## Route guide (works without any card / paid account)
When a user asks how to reach an institute, the bot builds a route card:
- **Exact-name Google Maps buttons** (transit, walking, driving, "find the institute"). They are built from the institute's NAME and the user's start, so Google Maps itself finds the exact place and shows bus numbers, stops and timings. No API key needed.
- **Nearest metro station, railway station and bus stop** to the institute (straight-line distance) from OpenStreetMap (Overpass).
- **Taxi distance/time estimate** (OSRM, without live traffic) and an **Uber** button with pickup/drop coordinates when the institute's exact location is known. Ola opens its booking page (pre-filled only if `OLA_XAPP_TOKEN` is set).
- Place lookup uses OpenStreetMap (Nominatim) when Google is not available. If only the institute's area is found, the card says so and cab buttons are hidden instead of guessing.
- The start can be typed ("Howrah station থেকে ...") or taken from the 📍 button (browser location, sent only while it is switched on).
- The bot never invents bus numbers: in this mode it tells users to use the Google Maps button for them.

**Automatic upgrade:** if a verified Google Cloud account/key is later set in `GOOGLE_PLACES_API_KEY` (with Places API (New) and Routes API enabled), the bot automatically uses Google Places, Google nearby stations and Google transit routes (bus numbers, steps) in the card. If Google refuses a request (account not verified, quota used up), the bot silently falls back to OpenStreetMap for the next 10 minutes.

Optional environment variables: `OSM_CONTACT` (your e-mail or site, sent in the User-Agent as OpenStreetMap's policy asks), `ROUTES_ENABLED=0` (turn the route guide off), `ROUTES_INCLUDE_DRIVE=0` (skip the taxi estimate), `OSRM_ENABLED=0`, `UBER_CLIENT_ID`, `OLA_XAPP_TOKEN`, `NOMINATIM_URL`, `OVERPASS_URL`, `OSRM_URL` (to point at your own servers).

Fair use: the public Nominatim, Overpass and OSRM servers are free but limited (Nominatim: 1 request/second; Overpass: about 10,000 requests/day). The bot caches results and throttles itself. For heavy traffic, run your own instances or use a paid geocoder. Map data (c) OpenStreetMap contributors (ODbL); the route card shows this credit when OSM data is used.

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
  - `GOOGLE_PLACES_API_KEY` = optional. Needs a verified Google Cloud account. Without it the app uses OpenStreetMap and web search.
  - `PLACES_INCLUDE_CONTACT` = `1` to also fetch phone and website (higher Google billing tier). Default `0`.
  - `GROQ_MODEL` = `openai/gpt-oss-120b` (optional; keep current model unless you intentionally change it)
  - `SECRET_KEY` is no longer needed (Flask sessions are not used).

Do not enable Flask debug mode in production. After deployment, test:
- `https://gemini-course-bot.onrender.com/health`
- Bengali: `হুগলিতে Montessori training institute খুঁজুন`
- English: `Find vocational training institutes in Howrah and give official sources`
- Hindi: `पश्चिम बंगाल में नर्सिंग कोर्स के लिए संस्थान खोजें`
- Route: `Howrah station से इस संस्थान तक कैसे जाएँ?` (the bot should ask for a destination if unclear)

## Rollback
If the new deployment fails, use GitHub's commit history to revert to the previously recorded working commit, or restore the backup copy. Do not delete the old working source until the new deployment passes testing.

## v6: official website reading, reviews link, profile answers
- If the user pastes the institute's website link, or the Google Places entry / a matching search result has one, the bot reads the home page plus up to 3 key pages (courses, contact, admission, placement...) and labels it "the institute's own claims". It checks robots.txt, allows only public http/https sites (ports 80/443), limits size, and skips social/aggregator sites. Set `SITE_FETCH_ENABLED=0` to turn it off.
- Institute names are kept exactly as typed (no transliteration).
- Place cards get a "⭐ See reviews on Google Maps" link (no key needed). The bot never invents ratings or review text.
- Answers follow an institute-profile layout and name the regulator/board to verify for regulated courses.

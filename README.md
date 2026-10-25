# Second Chance (phone app)

Installable web app (PWA). A GitHub Action runs the model every 6 hours and writes `docs/data.json`; the phone app is static HTML that reads it. No server, no cost.

## Deploy (about 10 minutes, free)
1. Create a **public** GitHub repo and upload everything in this folder (keep the `.github` folder).
2. Repo **Settings > Pages**: Source = "Deploy from a branch", Branch = `main`, folder = `/docs`. Save.
3. Repo **Actions** tab > "Refresh data" > **Run workflow** once (confirms the schedule works; `docs/data.json` is already included).
4. On your phone open `https://<your-username>.github.io/<repo-name>/`
   - iPhone: Safari > Share > Add to Home Screen
   - Android: Chrome > menu > Install app

## How updates work
- `export_data.py` imports `engine.py`, finds the newest nflverse season, retrains, writes `docs/data.json`. New seasons roll forward automatically.
- Roster is stored in the phone's local storage (private to that device, never uploaded). Install to the home screen so iOS doesn't clear it.
- GitHub pauses scheduled workflows after 60 days of repo inactivity; the data commits normally keep it alive, but if it's ever paused, click "Enable" in the Actions tab.

## Run locally
    pip install -r requirements.txt
    python export_data.py
    cd docs && python -m http.server 8000     # open http://localhost:8000

## Files
- `engine.py` data pull, features, models (same engine as the Streamlit version)
- `export_data.py` writes `docs/data.json`
- `docs/` the app: `index.html`, `manifest.webmanifest`, `sw.js` (offline cache), `icons/`
- `.github/workflows/refresh.yml` the 6-hourly refresh

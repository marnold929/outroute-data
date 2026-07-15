# outroute-data — OutRoute's own daily rankings pipeline

This repo IS your data company. Every morning it pulls **free** sources, runs **your own
ranking model**, and publishes a fresh `players.json` that the OutRoute iOS app
downloads on launch — no App Store updates, no data licenses, no server bills.

## How it works

```
 6:15am ET daily (GitHub Actions, free)
        │
        ▼
 Sleeper API ──────────┐   free: rosters, injuries, depth charts, trending adds
 FFC ADP API ──────────┤   free: live market ADP from thousands of real mock drafts
 manual_overrides.json ┘   your research layer: news notes + rank nudges
        │
        ▼
 pipeline/model.py  ←──── YOUR ranking model (injury penalties, depth-chart checks,
        │                 momentum, gap-based tiers, superflex re-rank)
        ▼
 docs/players.json  ←──── published at a public URL the app fetches
```

**Why this is legally clean to sell:** ADP is factual market data from a free public API,
Sleeper's API is free and public, and the *rankings are computed by your own model* —
you're not redistributing anyone's expert content.

## One-time setup (10 minutes)

1. Create a free account at github.com (if you don't have one).
2. Create a new **public** repository named `outroute-data`.
3. Upload everything in this folder to it (drag-and-drop works on github.com, or `git push`).
4. On the repo page: **Settings → Pages → Source: "Deploy from a branch" → Branch: `main`, folder: `/docs` → Save.**
5. Go to the **Actions** tab → enable workflows → open "Daily rankings build" → **Run workflow** to do your first build.
6. Your feed is now live at:
   `https://YOUR-USERNAME.github.io/outroute-data/players.json`
   (also works instantly without Pages at
   `https://raw.githubusercontent.com/YOUR-USERNAME/outroute-data/main/docs/players.json`)
7. Paste that URL into the OutRoute app: **League tab → Data feed → feed URL**.

That's it. The workflow runs every morning automatically, forever, for free.

## The manual/agent research layer

`pipeline/manual_overrides.json` is where human or AI research lands:

```json
{
  "news":       { "Player Name": "Suspension appeal pending — draft with caution" },
  "rank_nudge": { "Player Name": -8 },
  "exclude":    [ "Player Name" ]
}
```

- **news** — a note shown on the player's card and factored into compare confidence.
- **rank_nudge** — move a player up (+) or down (−) that many rank spots vs your model.
- **exclude** — remove someone entirely (holdouts, retirements the APIs haven't caught).

Edit it on github.com directly — any push rebuilds the data automatically. Ask Claude to
"do a news sweep for OutRoute" anytime; it can research current storylines and give you
an updated overrides file to paste in.

## Maintenance calendar

- **Each July:** bump `SEASON_YEAR` in `pipeline/build.py`, refresh `pipeline/byes.json`
  with the new season's bye weeks (ask Claude), clear stale overrides.
- **That's all.** Everything else is automatic.

## Local dev

```bash
python pipeline/build.py --fixtures   # offline test with bundled fixtures
python pipeline/build.py              # real build (needs internet)
```

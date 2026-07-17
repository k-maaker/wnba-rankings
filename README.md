# WNBA Betting Market Rankings

A self-hosted recreation of inpredictable's WNBA betting market rankings
(methodology from the July 2016 post), with two fixes over the original page:
sparklines are inline SVG (the original used the now-dead Google Image Charts
API, which is why its trend graphs are broken), and GOU is computed correctly.

## What it does

Every run it:

1. Pulls scores, point spreads, and over/unders for every WNBA game from
   ESPN's public scoreboard API and accumulates them in `data/games.csv`
   (lines are never overwritten with blanks, so closing lines persist).
2. Fits a weighted least-squares regression on line-implied and actual team
   points: `pts = mu + o_team - d_opp ± hca/2`, with recency weights
   `1/(elapsed_games + 0.25)` for market lines and `1/(elapsed_games + 3.5)`
   for game results — the exact weight forms from the original methodology.
   Home court advantage is estimated inside the regression rather than fixed.
3. Derives GPF (= oGPF + dGPF), GOU (= 2·mu + oGPF − dGPF), simple SRS, and
   W-L, snapshots the day's ratings into `data/history.csv`, and renders
   `docs/index.html` with rank moves vs. last week and 30-day GPF sparklines.

## One-time setup (~5 minutes)

1. Create a GitHub repo and push these files.
2. Repo Settings → Pages → Source: **Deploy from a branch**, branch `main`,
   folder `/docs`. Your rankings will live at
   `https://<user>.github.io/<repo>/`.
3. Repo Settings → Actions → General → Workflow permissions →
   **Read and write permissions** (so the bot can commit data).
4. Actions tab → "Update WNBA rankings" → **Run workflow** once manually.
   The first run backfills the season automatically (empty data file
   triggers a full-season scan from `WNBA_SEASON_START`, default 2026-05-01).

After that it runs itself every morning at 5:00 AM ET (cron `0 9 * * *` UTC).

## Local use

```bash
pip install numpy pandas
python wnba_rankings.py             # fetch, model, render docs/index.html
python wnba_rankings.py --backfill  # re-scan the whole season
python wnba_rankings.py --no-fetch  # recompute from stored data only
python wnba_rankings.py --test      # synthetic-data sanity check
```

## Notes and caveats

- **Line coverage**: ESPN retains odds on past scoreboard pages inconsistently.
  The model degrades gracefully — games without lines still contribute via the
  (down-weighted) results rows — but the rankings get sharper as the daily runs
  accumulate genuine closing lines. If you want deeper line history, The Odds
  API's paid tier has historical WNBA closers you could backfill into
  `data/games.csv` (columns: `game_id,date,home,away,home_score,away_score,
  spread_home,total`; `spread_home` negative = home favored).
- **LstWk / sparklines** need history, so they populate over the first week of
  runs (backfill can't reconstruct what the model would have said on past days
  without past-day lines).
- **DST**: cron is fixed at 09:00 UTC. That's 5 AM ET all season; in winter it
  would be 4 AM ET, which for an off-season no-op doesn't matter.
- Offense/defense are on a points-per-game basis, not per-possession — pace
  can't be separated out of spread + total alone (same limitation as the
  original).

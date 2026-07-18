#!/usr/bin/env python3
"""Download the BigDataBall Dropbox folder (as a zip) and copy the newest
.xlsx feed into data/bigdataball/, replacing whatever was there before.

Reads the folder's shared link from the DROPBOX_FOLDER_URL env var. Never
raises on failure — a bad or missing link just means this run falls back to
whatever .xlsx is already committed in data/bigdataball/, so a Dropbox
hiccup never breaks the site, it just delays fresh odds by a day.
"""
import os
import shutil
import sys
import urllib.request
import zipfile

DEST_DIR = "data/bigdataball"
TMP_ZIP = "/tmp/bdb.zip"
TMP_DIR = "/tmp/bdb_extracted"


def to_direct_download(url: str) -> str:
    url = url.replace("?dl=0", "").replace("&dl=0", "")
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}dl=1"


def main() -> None:
    url = os.environ.get("DROPBOX_FOLDER_URL", "").strip()
    if not url:
        print("No DROPBOX_FOLDER_URL set — skipping Dropbox pull.")
        return

    url = to_direct_download(url)
    print("Downloading Dropbox folder...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(TMP_ZIP, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:  # noqa: BLE001
        print(f"!! Download failed: {e} — keeping existing file.")
        return

    size = os.path.getsize(TMP_ZIP)
    print(f"Downloaded {size} bytes.")

    if not zipfile.is_zipfile(TMP_ZIP):
        print("!! Downloaded file is not a valid zip — the Dropbox link may "
              "be wrong, expired, or point at a single file instead of a "
              "folder. First 300 bytes for debugging:")
        with open(TMP_ZIP, "rb") as f:
            print(f.read(300))
        return

    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        with zipfile.ZipFile(TMP_ZIP) as z:
            z.extractall(TMP_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"!! Failed to extract zip: {e} — keeping existing file.")
        return

    candidates = []
    for root, _, files in os.walk(TMP_DIR):
        for fn in files:
            if fn.lower().endswith(".xlsx"):
                candidates.append(os.path.join(root, fn))

    if not candidates:
        print("!! No .xlsx found inside the downloaded zip — keeping existing file.")
        return

    feed = [c for c in candidates
            if "team-feed" in c.lower() or "wnba" in c.lower()]
    latest = sorted(feed or candidates)[-1]

    os.makedirs(DEST_DIR, exist_ok=True)
    for old in os.listdir(DEST_DIR):
        if old.lower().endswith((".xlsx", ".xlsm")):
            os.remove(os.path.join(DEST_DIR, old))
    shutil.copy(latest, DEST_DIR)
    print(f"Using: {os.path.basename(latest)}")


if __name__ == "__main__":
    main()
    sys.exit(0)  # never fail the job over a Dropbox hiccup

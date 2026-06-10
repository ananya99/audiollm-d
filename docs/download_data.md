# How to download the DementiaBank Pitt Corpus

Prerequisites: You need a TalkBank account with access to the DementiaBank dataset. Request access at talkbank.org if you don't have one.

> **Important:** On `media.talkbank.org`, the base URL for each file (e.g. `002-0.mp3`) serves an 11-byte HTML redirect stub. The actual audio content is at the `?f=save` URL (e.g. `002-0.mp3?f=save`). The commands below keep the `?f=save` files and clean up stubs afterward.

## Step 1 — Get your session cookie

1. Open Chrome/Firefox and log in at [https://media.talkbank.org](https://media.talkbank.org)
2. Install the browser extension **"Get cookies.txt LOCALLY"**
3. While on `media.talkbank.org`, click the extension and export cookies — save the file as `~/Downloads/media.talkbank.org_cookies.txt`

> The session cookie expires after some time. If a download stops authenticating mid-way, repeat this step and re-run.

## Step 2 — Run the download

### Option A — Full Pitt corpus at once

Downloads everything under `/dementia/English/Pitt/` in one command. Safe to re-run — `--no-clobber` skips already-downloaded files.

```bash
COOKIES=~/Downloads/media.talkbank.org_cookies.txt

wget \
  --recursive \
  --no-parent \
  --no-host-directories \
  --cut-dirs=3 \
  --reject "*.html,*.tmp,*.js" \
  --no-clobber \
  --continue \
  --load-cookies "$COOKIES" \
  --keep-session-cookies \
  -P data/dementia/pitt \
  "https://media.talkbank.org/dementia/English/Pitt/"
```

### Option B — Per subfolder (use if cookies expire mid-download)

```bash
COOKIES=~/Downloads/media.talkbank.org_cookies.txt

for group in Dementia Control; do
  for task in cookie fluency recall sentence; do
    echo ">>> Downloading $group/$task"
    wget \
      --recursive \
      --no-parent \
      --no-host-directories \
      --cut-dirs=3 \
      --reject "*.html,*.tmp,*.js" \
      --no-clobber \
      --continue \
      --load-cookies "$COOKIES" \
      --keep-session-cookies \
      -P data/dementia/pitt \
      "https://media.talkbank.org/dementia/English/Pitt/$group/$task/"
  done
done

for subfolder in noiseRemoval_ADReSS Dementia_noise Control_noise; do
  echo ">>> Downloading 0extra/$subfolder"
  wget \
    --recursive \
    --no-parent \
    --no-host-directories \
    --cut-dirs=3 \
    --reject "*.html,*.tmp,*.js" \
    --no-clobber \
    --continue \
    --load-cookies "$COOKIES" \
    --keep-session-cookies \
    -P data/dementia/pitt \
    "https://media.talkbank.org/dementia/English/Pitt/0extra/$subfolder/"
done
```

If a specific subfolder fails, re-export cookies and re-run just that subfolder:

```bash
wget \
  --recursive \
  --no-parent \
  --no-host-directories \
  --cut-dirs=3 \
  --reject "*.html,*.tmp,*.js" \
  --no-clobber \
  --continue \
  --load-cookies ~/Downloads/media.talkbank.org_cookies.txt \
  --keep-session-cookies \
  -P data/dementia/pitt \
  "https://media.talkbank.org/dementia/English/Pitt/Dementia/cookie/"
```

## Step 3 — Rename real files and remove stubs

wget saves the actual audio/data content as `filename.ext?f=save` and also creates an 11-byte HTML stub at `filename.ext`. Run this after every download to fix the filenames:

```bash
# Rename ?f=save → correct filename, skipping any target that is a directory
find data/dementia/pitt -name '*\?f=save' | while IFS= read -r f; do
  target="$(echo "$f" | sed 's/?f=save$//')"
  if [ -d "$target" ]; then
    echo "SKIPPING (target is a directory): $f"
  elif [ -f "$target" ] && [ "$(wc -c < "$target")" -gt 100 ]; then
    echo "SKIPPING (real file already exists): $target"
  else
    mv "$f" "$target"
  fi
done

# Delete ?f=open duplicates
find data/dementia/pitt -name '*\?f=open' -delete

# Delete any remaining 11-byte audio stubs
find data/dementia/pitt \( -name '*.wav' -o -name '*.mp3' \) -size -100c -delete
```

## Result

Files will be organized as:

```text
data/dementia/pitt/
  Dementia/
    cookie/         ← files named e.g. 001-0.mp3?f=save
      0wav/         ← noise-removed WAVs: 001-0.wav?f=save
    fluency/
    recall/
    sentence/
  Control/
    cookie/
      0wav/
    fluency/
    recall/
    sentence/
  0extra/
    noiseRemoval_ADReSS/   ← MATLAB spectral subtraction scripts
    Dementia_noise/        ← noise estimate WAVs for Dementia
    Control_noise/         ← noise estimate WAVs for Control
```

The `--no-clobber` flag makes every command safe to re-run — already-downloaded files are skipped.

#!/usr/bin/env bash
# Run MFA forced alignment for AudioLLM-D disfluency corpus.
#
# Prerequisites:
#   conda install -c conda-forge montreal-forced-aligner
#   (MFA from PyPI is missing kalpy — the compiled Kaldi backend — so use conda.)
#
# Or on the RCP cluster, if MFA is available via module:
#   module load mfa   (or activate the shared conda env that has it)
#
# Usage:
#   bash scripts/run_mfa_alignment.sh [--adapt] [--pilot N]
#
#   --adapt     Run mfa adapt before final alignment (recommended for dysarthric speech).
#   --pilot N   Only align the first N clips (sanity check before full run).
#
# Outputs:
#   mfa/corpus/          .lab + .wav symlinks
#   mfa/merged_dict.txt  system dict + custom entries merged
#   mfa/alignments/      .TextGrid per clip (raw MFA output)
#   mfa/adapted_model/   adapted acoustic model (if --adapt)
#   mfa/processed/       _alignment.json + _aligned.txt per clip

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIOLLMD_DIR="$REPO_ROOT/AudioLLM-D"
MFA_DIR="$REPO_ROOT/mfa"
MERGED_DIR="$REPO_ROOT/merged/merged_data_processed"

DO_ADAPT=false
PILOT_N=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapt) DO_ADAPT=true; shift ;;
        --pilot) PILOT_N="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1 — corpus prep
# ---------------------------------------------------------------------------
echo "[1/5] Preparing MFA corpus..."
python "$AUDIOLLMD_DIR/scripts/prepare_mfa_input.py" \
    --audio-dir "$MERGED_DIR/audio" \
    --transcript-dir "$MERGED_DIR/transcripts" \
    --output-dir "$MFA_DIR/corpus"

# Pilot mode: only keep the first N clips.
if [[ -n "$PILOT_N" ]]; then
    echo "[pilot] Keeping only $PILOT_N clips for pilot run."
    PILOT_DIR="$MFA_DIR/corpus_pilot"
    mkdir -p "$PILOT_DIR"
    find "$MFA_DIR/corpus" -name "*.lab" | sort | head -n "$PILOT_N" | while read -r f; do
        stem="$(basename "${f%.lab}")"
        cp "$f" "$PILOT_DIR/"
        cp -P "$MFA_DIR/corpus/${stem}.wav" "$PILOT_DIR/" 2>/dev/null || \
            cp "$MFA_DIR/corpus/${stem}.wav" "$PILOT_DIR/"
    done
    CORPUS_DIR="$PILOT_DIR"
    ALIGN_OUT="$MFA_DIR/alignments_pilot"
else
    CORPUS_DIR="$MFA_DIR/corpus"
    ALIGN_OUT="$MFA_DIR/alignments"
fi

# ---------------------------------------------------------------------------
# Step 2 — merge pronunciation dictionaries
# ---------------------------------------------------------------------------
echo "[2/5] Merging pronunciation dictionaries..."
MFA_DICT_PATH="$(mfa model inspect dictionary english_us_arpa 2>/dev/null | grep -m1 'path:' | awk '{print $2}')" || true
if [[ -z "$MFA_DICT_PATH" ]]; then
    # Fallback: try common MFA model storage locations.
    for candidate in \
        "$HOME/.local/share/montreal-forced-aligner/pretrained_models/dictionary/english_us_arpa.dict" \
        "$HOME/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict" \
        "$CONDA_PREFIX/share/mfa/pretrained_models/dictionary/english_us_arpa.dict"; do
        if [[ -f "$candidate" ]]; then
            MFA_DICT_PATH="$candidate"
            break
        fi
    done
fi

if [[ -z "$MFA_DICT_PATH" ]]; then
    echo "[warn] Could not find english_us_arpa.dict; trying to download..."
    mfa model download dictionary english_us_arpa
    MFA_DICT_PATH="$(mfa model inspect dictionary english_us_arpa | grep -m1 'path:' | awk '{print $2}')"
fi

# Strip comment lines from custom dict, then merge.
grep -v '^#' "$MFA_DIR/custom_dict.txt" > "$MFA_DIR/custom_dict_nocomments.txt"
cat "$MFA_DICT_PATH" "$MFA_DIR/custom_dict_nocomments.txt" > "$MFA_DIR/merged_dict.txt"
echo "  Merged dict: $MFA_DIR/merged_dict.txt"

# ---------------------------------------------------------------------------
# Step 3 — optional acoustic model adaptation
# ---------------------------------------------------------------------------
if $DO_ADAPT; then
    echo "[3/5] Adapting acoustic model (mfa adapt)..."
    mfa adapt \
        "$CORPUS_DIR" \
        "$MFA_DIR/merged_dict.txt" \
        english_us_arpa \
        "$MFA_DIR/adapted_model" \
        --clean \
        --num_iterations 40
    ACOUSTIC_MODEL="$MFA_DIR/adapted_model"
    echo "  Adapted model: $ACOUSTIC_MODEL"
else
    echo "[3/5] Skipping adaptation (pass --adapt to enable)."
    ACOUSTIC_MODEL="english_us_arpa"
fi

# ---------------------------------------------------------------------------
# Step 4 — forced alignment
# ---------------------------------------------------------------------------
echo "[4/5] Running forced alignment..."
mfa align \
    "$CORPUS_DIR" \
    "$MFA_DIR/merged_dict.txt" \
    "$ACOUSTIC_MODEL" \
    "$ALIGN_OUT" \
    --clean \
    --beam 100 \
    --retry_beam 400 \
    --output_format long_textgrid

echo "  TextGrids written to: $ALIGN_OUT"

# ---------------------------------------------------------------------------
# Step 5 — post-process TextGrids
# ---------------------------------------------------------------------------
echo "[5/5] Post-processing TextGrids..."
python "$AUDIOLLMD_DIR/scripts/postprocess_mfa.py" \
    --textgrid-dir "$ALIGN_OUT" \
    --transcript-dir "$MERGED_DIR/transcripts" \
    --output-dir "$MFA_DIR/processed"

echo ""
echo "Done. Inspect a few outputs:"
echo "  ls $MFA_DIR/processed/ | head -10"
echo "  cat $MFA_DIR/processed/<stem>_aligned.txt"
echo "  python -c \"import json; print(json.dumps(json.load(open('$MFA_DIR/processed/<stem>_alignment.json')), indent=2))\""

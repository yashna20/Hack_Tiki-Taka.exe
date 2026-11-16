#!/bin/sh
# Download the SkillCorner open data (MIT licence) used as extra practice matches by v6.
#
#   sh v6/download_skillcorner.sh
#
# The repository keeps its tracking files in Git LFS. Without git-lfs a clone only holds small
# pointer files, so each one is replaced by the real file from GitHub's media server and checked
# against the SHA-256 written in its pointer. Everything lands in data/external/ (gitignored).
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/data/external/skillcorner-opendata"
COMMIT=4340d274572876239c154c90bc507a9b3250a656
MEDIA="https://media.githubusercontent.com/media/SkillCorner/opendata/$COMMIT"

if [ ! -d "$DEST/.git" ]; then
    git clone https://github.com/SkillCorner/opendata "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout -q "$COMMIT"

cd "$DEST"
for pointer in data/matches/*/*_tracking_extrapolated.jsonl; do
    if ! head -c 100 "$pointer" | grep -q "git-lfs.github.com/spec"; then
        echo "ok (already downloaded)  $pointer"
        continue
    fi
    want=$(sed -n 's/^oid sha256://p' "$pointer")
    # HTTP/1.1 + resume: large files from the media server sometimes drop mid-stream over HTTP/2
    curl -fsSL --http1.1 --retry 10 --retry-all-errors --retry-delay 5 -C - \
        -o "$pointer.part" "$MEDIA/$pointer"
    got=$(shasum -a 256 "$pointer.part" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "CHECKSUM MISMATCH $pointer" >&2
        rm -f "$pointer.part"
        exit 1
    fi
    mv "$pointer.part" "$pointer"
    echo "downloaded + verified  $pointer"
done

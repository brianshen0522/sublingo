#!/usr/bin/env bash
# Extract subtitles from an MKV file
# Usage: ./extract_subs.sh <input.mkv> [stream_index] [output.srt]

set -euo pipefail

INPUT="${1:?Usage: $0 <input.mkv> [stream_index] [output.srt]}"
STREAM="${2:-0}"
OUTPUT="${3:-${INPUT%.*}.srt}"

# List all subtitle streams
echo "Available subtitle streams:"
ffprobe -v error -select_streams s -show_entries stream=index:stream_tags=language,title -of csv=p=0 "$INPUT"
echo ""

echo "Extracting subtitle stream $STREAM -> $OUTPUT"
ffmpeg -y -i "$INPUT" -map "0:s:${STREAM}" -c:s srt "$OUTPUT"
echo "Done: $OUTPUT"

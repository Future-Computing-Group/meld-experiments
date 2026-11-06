#!/bin/sh
# Fetch the pinned Caura tree into vendor/caura/src (see PIN). Refuses on a sha256 mismatch.
set -eu
cd "$(dirname "$0")"
SHA=fd1da78b563b21f357fe5a7060a79954c9df94c0
SUM=d8347126b147b839fecb18f11416c11b4255058002e5aed1e6d6c6334698b4d3
TGZ=caura-${SHA%"${SHA#????????}"}.tar.gz
[ -f "$TGZ" ] || curl -sSfL -o "$TGZ" "https://codeload.github.com/caura-ai/caura/tar.gz/$SHA"
GOT=$(shasum -a 256 "$TGZ" 2>/dev/null | cut -d' ' -f1 || sha256sum "$TGZ" | cut -d' ' -f1)
[ "$GOT" = "$SUM" ] || { echo "sha256 mismatch: $GOT (want $SUM); compare the excerpt file hashes instead" >&2; exit 1; }
rm -rf src && mkdir src && tar -xzf "$TGZ" -C src --strip-components=1
echo "vendor/caura/src = caura-ai/caura@$SHA"

#!/usr/bin/env bash
# Local preview at http://localhost:4000 using Docker (no system Ruby required).
# Native alternative: sudo apt install ruby-dev build-essential zlib1g-dev,
# then `bundle install && bundle exec jekyll serve --livereload`.
set -euo pipefail

cd "$(dirname "$0")"

# Runs as the host user so _site/ and vendor/ aren't left root-owned.
exec docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e BUNDLE_PATH=/site/vendor/bundle \
  -v "$PWD":/site -w /site \
  -p 4000:4000 -p 35729:35729 \
  ruby:3.3 \
  bash -c "bundle install --quiet && bundle exec jekyll serve --host 0.0.0.0 --livereload --force_polling $*"

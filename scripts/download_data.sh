#!/usr/bin/env bash
# Downloads RuReviews (Apache-2.0) from the authors' repository.
set -euo pipefail
mkdir -p data/raw
curl -sSL -o data/raw/rureviews.csv \
    https://raw.githubusercontent.com/sismetanin/rureviews/master/women-clothing-accessories.3-class.balanced.csv
wc -l data/raw/rureviews.csv

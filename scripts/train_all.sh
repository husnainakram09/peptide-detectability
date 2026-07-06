#!/usr/bin/env bash
set -e
python -m src.data_prep
python -m src.train --model baseline
python -m src.train --model cnn
python -m src.train --model esm2
python -m src.benchmark
echo "Done. See reports/benchmark.md"

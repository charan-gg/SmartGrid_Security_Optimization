#!/bin/bash
# Steps 1-3 + E1-E4 for the 50% implementation review.
set -o pipefail
cd "$(dirname "$0")"
P=./.venv/bin/python
echo "START $(date -u +%FT%TZ)  backend check:"
$P -c "from src.models import backend_name; print(backend_name())"
rm -rf outputs/cache
$P step1_profile.py                            > outputs/step1.log 2>&1 && echo "step1 done $(date -u +%T)"
$P step2_features.py                           > outputs/step2.log 2>&1 && echo "step2 done $(date -u +%T)"
$P step3_split_and_baseline.py                 > outputs/step3.log 2>&1 && echo "step3 done $(date -u +%T)"
$P step4_run_experiments.py --only e1 e2 e3 e4 > outputs/step4.log 2>&1 && echo "step4 done $(date -u +%T)"
echo "FINISHED $(date -u +%FT%TZ)"

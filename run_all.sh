#!/bin/sh
# Runs the whole course in order. Trained models are saved in runs/ and reused on the next call.
set -e
cd "$(dirname "$0")"
python3 tests.py
for step in step1_notes step2_one_note step3_squeeze step4_four_bits step5_index step6_two_halves step7_bill; do
    echo
    echo "=== $step.py"
    python3 "$step.py"
done

#!/usr/bin/env python3
"""Reuse locked qualitative inference for the near-overhead v2 case selection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import infer_report_figure_cases as runner
from tools.select_report_figure_cases import digest, write
import json

if __name__ == '__main__':
    runner.OUT = ROOT / 'outputs/report_figures_swinl_v2'
    runner.main()
    receipt = runner.OUT / 'supplemental_predictions/run.json'
    meta = json.loads(receipt.read_text())
    meta['entrypoint'] = {'path': __file__, 'sha256': digest(__file__)}
    write(receipt, meta)

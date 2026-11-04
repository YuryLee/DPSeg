#!/usr/bin/env python3
"""Use the unchanged final auditor on the declared hardware recovery root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import finish_course_depth_backbone as watcher

if __name__ == '__main__':
    watcher.OUT = ROOT/'reports/course_depth_backbone_v1_gpu02'
    watcher.main()

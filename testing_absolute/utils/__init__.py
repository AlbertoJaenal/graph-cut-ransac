"""Absolute pose estimation utilities and solver definitions.

Provides solvers, paths, feature extraction, and evaluation tools for testing
absolute pose estimation methods on benchmark datasets.

Key modules:
- absolute_defs: Solver enumeration
- absolute_solvers: RANSAC verification backends
- absolute_estimators: GC-RANSAC pose estimator
- utils: SfM model creation and correspondence handling
- geometry_utils: Camera transformations and normal estimation
- extract_scale_ori: SIFT scale/orientation extraction
- eval_utils: Evaluation and error metrics
- utils_cambridge: Cambridge dataset utilities
"""
import sys
import os
from pathlib import Path

RANSAC_DIR = Path(os.path.abspath(__file__)).parent.parent.parent.absolute()
sys.path.insert(0, str(RANSAC_DIR / "build"))
import pygcransac

DATA_DIR = "/home/alberto/work/data/"

# Import main interfaces
from .absolute_estimators import main_gcloc
from .absolute_defs import Solvers
from .utils import create_model_or_retrieve
from .eval_utils import evaluate_generic, evaluate_ftim
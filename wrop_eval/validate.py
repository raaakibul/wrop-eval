"""
validate.py — correlate per-model OPS (fully automatic) against human
blind pairwise Elo ratings (expensive, slow, one-off).  
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy import stats
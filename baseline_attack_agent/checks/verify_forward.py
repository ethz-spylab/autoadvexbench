import os
import sys
import numpy as np
from PIL import Image


def run_original_model_forward(fp):
    import subprocess
    try:
        result = subprocess.run(
            ['python3', 'forward.py', fp],
            capture_output=True,
            text=True,
            cwd='/usr/src/app'
        )
    except:
        result = subprocess.run(
            ['python3', 'forward.py', fp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd='/usr/src/app'
        )
    if result.returncode != 0:
        print("Error running forward.py:", result.stderr)
        raise RuntimeError("forward.py execution failed")
    labels = result.stdout
    labels = labels.split("Labels\n")[-1]
    labels = list(map(int, labels.split()))
    labels = np.array(labels)
    assert len(labels) == 1000, "Processed fewer labels than expected"
    return labels

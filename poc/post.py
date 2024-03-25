import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np
import re
import os
import math

dtype = np.dtype("float64")
nfields = 4

for path in sys.argv[1:]:
    size = os.path.getsize(path)
    ny = math.isqrt(size // dtype.itemsize // nfields // 4)
    nx = 4 * ny
    u, v, rho, vort = np.memmap(path, dtype=dtype, shape=(nfields, ny, nx))
    plt.figure(figsize=(20, 5), dpi=150, frameon=False)
    plt.axis("off")
    plt.axis("equal")
    plt.tight_layout()
    plt.imshow(vort, matplotlib.cm.jet, vmin=-0.01, vmax=0.01)
    png = re.sub("\.raw$", "", path) + ".png"
    print(png)
    plt.savefig(png, bbox_inches=0)
    plt.close()

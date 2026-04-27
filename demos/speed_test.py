#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import pywt
import time

# =========================================================
# db4 wavelet filters — pulled directly from PyWavelets
# =========================================================
_w = pywt.Wavelet('db4')
FILTER_LEN = _w.dec_len  # 8

# Flip for true convolution: PyWavelets convolves with the filter,
# but torch.nn.functional.conv1d/conv2d does cross-correlation.
# Flipping converts cross-correlation into convolution.
h = torch.tensor(_w.dec_lo, dtype=torch.float32).flip(0)  # low-pass
g = torch.tensor(_w.dec_hi, dtype=torch.float32).flip(0)  # high-pass

# =========================================================
# PyWavelets reference implementation
# =========================================================
def dwt2_pywt(img):
    LL, (LH, HL, HH) = pywt.dwt2(img, wavelet='db4', mode='reflect')
    return LL, LH, HL, HH

# =========================================================
# PyTorch wavelet transform (PyWavelets-aligned)
# =========================================================
def dwt2_torch(x):
    """
    x : (H, W) float32 tensor
    Returns LL, LH, HL, HH matching pywt.dwt2(..., mode='reflect') exactly.

    Key details to match PyWavelets behaviour:
    -----------------------------------------
    1. FILTER FLIP: pywt convolves; F.conv2d cross-correlates.
       Flipping dec_lo / dec_hi makes them equivalent.

    2. PADDING: pywt 'reflect' mode pads left = filter_len-2 = 6,
       right = filter_len-1 = 7, on each axis (asymmetric).

    3. AXIS/SUBBAND CONVENTION (from pywt docs):
       pywt treats axis-0 as "horizontal" (unusual for images):
         LH = high-pass on axis-0 (rows), low-pass on axis-1 (cols)
         HL = low-pass on axis-0 (rows), high-pass on axis-1 (cols)
       So in torch.outer(frow, fcol):
         pywt LH  <->  frow=g (high), fcol=h (low)
         pywt HL  <->  frow=h (low),  fcol=g (high)
    """
    dev = x.device
    h_loc = h.to(dev)
    g_loc = g.to(dev)

    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Asymmetric reflect padding: left=F-2=6, right=F-1=7 on each axis
    x = F.pad(x, (FILTER_LEN - 2, FILTER_LEN - 1,   # cols: left, right
                   FILTER_LEN - 2, FILTER_LEN - 1),  # rows: top, bottom
               mode='reflect')

    def sep(frow, fcol):
        """Separable 2-D strided convolution."""
        filt = torch.outer(frow, fcol).unsqueeze(0).unsqueeze(0)
        return F.conv2d(x, filt, stride=2)[0, 0]

    LL = sep(h_loc, h_loc)  # low  × low
    LH = sep(g_loc, h_loc)  # high-row × low-col   (pywt "Horizontal detail")
    HL = sep(h_loc, g_loc)  # low-row  × high-col  (pywt "Vertical detail")
    HH = sep(g_loc, g_loc)  # high × high

    return LL, LH, HL, HH

# =========================================================
# Error metric (MSE)
# =========================================================
def mse(a, b):
    h_ = min(a.shape[0], b.shape[0])
    w_ = min(a.shape[1], b.shape[1])
    return np.mean((a[:h_, :w_] - b[:h_, :w_]) ** 2)

# =========================================================
# Benchmark function
# =========================================================
def benchmark(N=50, H=128, W=128, device="cuda"):

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

    py_times    = []
    torch_times = []
    errors      = []

    print("\n===== WAVELET BENCHMARK =====")
    print(f"Device: {dev}")
    print(f"Iterations: {N}, Image size: {H}x{W}\n")

    for i in range(N):
        img = np.random.randn(H, W).astype(np.float32)

        # PyWavelets
        t0 = time.time()
        LL1, LH1, HL1, HH1 = dwt2_pywt(img)
        py_times.append(time.time() - t0)

        # PyTorch
        x = torch.tensor(img, device=dev)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        LL2, LH2, HL2, HH2 = dwt2_torch(x)
        if use_cuda:
            torch.cuda.synchronize()
        torch_times.append(time.time() - t0)

        err = (mse(LL1, LL2.cpu().numpy()) +
               mse(LH1, LH2.cpu().numpy()) +
               mse(HL1, HL2.cpu().numpy()) +
               mse(HH1, HH2.cpu().numpy()))
        errors.append(err)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{N}")

    print("\n===== RESULTS =====")
    print(f"PyWavelets avg time : {np.mean(py_times):.6f} s")
    print(f"PyTorch avg time    : {np.mean(torch_times):.6f} s")
    print(f"Mean MSE difference : {np.mean(errors):.6e}")

# =========================================================
# run
# =========================================================
if __name__ == "__main__":
    benchmark(N=50000, H=128, W=128, device="cuda")

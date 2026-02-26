#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core functions for the Greedy Semi-Soft GoDec decomposition (GreBsmo).

References:
    Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank & Sparse Matrix
    Decomposition in Noisy Case", ICML 2011.

    Tianyi Zhou and Dacheng Tao, "Greedy Bilateral Sketch, Completion and
    Smoothing", AISTATS 2013.

Python implementation: J. Fattaccioli (Department of Chemistry, ENS)
Date: March 2020
"""

import numpy as np
import pywt
import scipy as sc
import tifffile as tif
import cv2 as cv


def DecomposeGoDec(
    filename: str,
    rank: int = 3,
    power: int = 5,
    tau: float = 7,
    tol: float = 0.001,
    k: int = 2,
    dynamicrange: int = 8,
    max_frames: int = 0,
) -> None:
    """
    Entry point: perform Low-Rank + Sparse + Noise decomposition on a video or TIFF stack.

    Parameters
    ----------
    filename : str
        Path to the input file (.tif, .tiff, or .avi).
    rank : int
        Upper bound on the rank of the low-rank component.
    power : int
        Power scheme order (>= 0). Higher values improve accuracy at the cost of runtime.
    tau : float
        Soft thresholding parameter for the sparse component.
    tol : float
        Convergence tolerance (relative residual).
    k : int
        Rank step size.
    dynamicrange : int
        Bit depth of the output TIFF stacks (8 or 16).
    max_frames : int
        Number of frames to process (0 = all frames).

    Output
    ------
    Four TIFF stacks written to the current directory:
        1-Original.tif   : original frames (or grayscale conversion of .avi)
        2-Background.tif : low-rank component (background)
        3-Sparse.tif     : sparse component (moving objects)
        4-Noise.tif      : residual noise component
    """
    StartingImage, NImage, HImage, WImage = _import_image(filename, max_frames)
    print("Image loading OK")

    FinalImage = _vectorize(StartingImage, NImage, HImage, WImage)
    print("Image vectorization OK")

    D, L, S = _GreGoDec(FinalImage, rank, tau, tol, power, k)
    G = D - L - S
    print("GoDec OK")

    _reconstruct(D, HImage, WImage, NImage, "1-Original.tif", dynamicrange)
    _reconstruct(L, HImage, WImage, NImage, "2-Background.tif", dynamicrange)
    print("Reconstruction low-rank OK")

    _reconstruct(S, HImage, WImage, NImage, "3-Sparse.tif", dynamicrange)
    print("Reconstruction sparse OK")

    _reconstruct(G, HImage, WImage, NImage, "4-Noise.tif", dynamicrange)
    print("Full process OK")


def _import_image(filename: str, max_frames: int):
    """
    Load a TIFF stack or AVI movie into a numpy array.

    Parameters
    ----------
    filename : str
        Path to a .tif/.tiff stack or .avi file.
    max_frames : int
        Maximum number of frames to load (0 = all).

    Returns
    -------
    Image : np.ndarray, shape (N, H, W)
    NImage, HImage, WImage : int
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext in ("tif", "tiff"):
        Image = tif.imread(filename).astype(float)

        if max_frames > 0:
            max_frames = min(max_frames, Image.shape[0])
            Image = Image[:max_frames]
        print("TIF stack loading OK")

    elif ext == "avi":
        Cap = cv.VideoCapture(filename)
        frame_count = int(Cap.get(cv.CAP_PROP_FRAME_COUNT))
        Width = int(Cap.get(cv.CAP_PROP_FRAME_WIDTH))
        Height = int(Cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        Frames = min(max_frames, frame_count) if max_frames > 0 else frame_count
        Temp = np.zeros((Frames, Height, Width))

        for framenumber in range(Frames):
            if (framenumber + 1) % 10 == 0:
                print(f"Loading frame {framenumber + 1}/{Frames}")
            ret, frame = Cap.read()
            if not ret:
                print("Could not read frame (stream end?). Stopping.")
                break
            Temp[framenumber] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        Cap.release()
        Image = Temp
        print("AVI loading OK")

    else:
        raise ValueError(f"Unsupported file extension '.{ext}'. Expected .tif, .tiff, or .avi.")

    NImage, HImage, WImage = Image.shape
    return Image, NImage, HImage, WImage


def _vectorize(Image: np.ndarray, NImage: int, HImage: int, WImage: int) -> np.ndarray:
    """
    Reshape a (N, H, W) image stack into a (N, H*W) 2D matrix for GoDec processing.
    """
    FinalImage = np.zeros((NImage, HImage * WImage))
    for i in range(NImage):
        FinalImage[i, :] = Image[i].reshape(HImage * WImage)
    return FinalImage


def _reconstruct(
    vector: np.ndarray,
    line: int,
    col: int,
    time: int,
    name: str,
    dynamicrange: int,
) -> np.ndarray:
    """
    Reshape a (N, H*W) matrix back to (N, H, W) and save as a TIFF stack.
    """
    image = np.zeros((time, line, col))
    for i in range(time - 1):
        image[i] = vector[i].reshape((line, col))
    _save(image, name, dynamicrange)


def _save(data: np.ndarray, path: str, dynamicrange: int) -> None:
    """
    Normalise and save a numpy array as an 8-bit or 16-bit TIFF stack.

    Parameters
    ----------
    data : np.ndarray
    path : str
        Output file path.
    dynamicrange : int
        Bit depth: 8 or 16.
    """
    data = ((data - np.amin(data)) / np.amax(data - np.amin(data))) * (2**dynamicrange - 1)

    if dynamicrange == 8:
        data = data.astype(np.uint8)
    elif dynamicrange == 16:
        data = data.astype(np.uint16)
    else:
        raise ValueError("dynamicrange must be 8 or 16.")

    tif.imsave(path, data)


def _GreGoDec(
    D: np.ndarray,
    rank: int,
    tau: float,
    tol: float,
    power: int,
    k: int,
):
    """
    Greedy Semi-Soft GoDec (GreBsmo) decomposition.

    Parameters
    ----------
    D : np.ndarray, shape (n, p)
        Input data matrix (frames × pixels).
    rank : int
        Upper bound on rank of the low-rank component L.
    tau : float
        Soft thresholding value for the sparse component S.
    tol : float
        Convergence tolerance (relative residual norm).
    power : int
        Number of power iterations per rank increment.
    k : int
        Rank step size.

    Returns
    -------
    D : np.ndarray  — input matrix (transposed back if needed)
    L : np.ndarray  — low-rank component
    S : np.ndarray  — sparse component
    """
    # Ensure n >= p (transpose if needed)
    if D.shape[0] < D.shape[1]:
        D = np.transpose(D)

    normD = np.linalg.norm(D)
    rankk = round(rank / k)
    error = np.zeros((rank * power, 1), dtype=float)

    # Initialise L via truncated SVD
    X, s, Y = sc.sparse.linalg.svds(D, k)
    s = s * np.identity(k)
    X = X.dot(s)
    L = X.dot(Y)

    # Initialise S via soft thresholding of residual
    S = pywt.threshold(D - L, tau, "soft")

    # Initial error
    T = D - L - S
    error[0] = np.linalg.norm(T) / normD

    iii = 1
    stop = False
    alf = 0

    for r in range(1, rankk + 1):
        r = r - 1
        rrank = rank
        est_rank = 1
        alf = 0
        increment = 1

        if iii == power * (r - 2) + 1:
            iii = iii + power

        for iter in range(1, power + 1):
            # Update X via QR decomposition of L*Y'
            X = abs(L.dot(np.transpose(Y)))
            X, R = np.linalg.qr(X)

            # Update Y and L
            Y = np.transpose(X).dot(L)
            L = X.dot(Y)

            # Update S via soft thresholding
            T = D - L
            S = pywt.threshold(T, tau, mode="soft")

            # Compute error
            T = T - S
            ii = iii + iter - 1
            error[ii] = np.linalg.norm(T) / np.linalg.norm(D)

            if error[ii] < tol:
                stop = True
                break

            if rrank != rank:
                rank = rrank
                if est_rank == 0:
                    alf = 0
                    continue

            # Adaptive step size adjustment
            ratio = error[ii] / error[ii - 1]
            X1, Y1, L1, S1, T1 = X, Y, L, S, T

            if ratio >= 1.1:
                increment = max(0.1 * alf, 0.1 * increment)
                X, Y, L, S, T = X1, Y1, L1, S1, T1
                error[ii] = error[ii - 1]
                alf = 0
            elif ratio > 0.7:
                increment = max(increment, 0.25 * alf)
                alf = alf + increment

            # Update L with step
            X1, Y1, L1, S1, T1 = X, Y, L, S, T
            L = L + (1 + alf) * T

            # Check stagnation and trim Y if needed
            if iter > 8:
                if np.mean(np.divide(error[ii - 7: ii], error[ii - 8])) > 0.92:
                    iii = ii
                    if Y.shape[1] - X.shape[0] >= k:
                        Y = Y[0: X.shape[0] - 1, :]
                    break

        if stop:
            break

        # Extend coreset
        if r + 1 < rankk:
            RR = np.random.randn(k, D.shape[0])
            v = RR.dot(L)
            Y = np.block([[Y], [v]])

    error = [error != 0]
    L = X.dot(Y)

    if D.shape[0] > D.shape[1]:
        L = np.transpose(L)
        S = np.transpose(S)

    return np.transpose(D), L, S

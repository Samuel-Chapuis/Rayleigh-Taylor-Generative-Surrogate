import numpy as np
import pylab as pyl

def rescale(f,a=0,b=1):
    """
    Rescale an array linearly to a given interval [a, b].

    This function normalizes the input array so that its minimum maps to `a`
    and its maximum maps to `b`. If the input array is constant, the function
    returns an array filled with `a`.

    Args:
        f (np.ndarray): Input array to rescale.
        a (float, optional): Lower bound of the target interval. Defaults to 0.
        b (float, optional): Upper bound of the target interval. Defaults to 1.

    Returns:
        np.ndarray: Rescaled array with values in the interval [a, b].
    """

    v = f.max() - f.min()
    g = (f - f.min()).copy()
    if v > 0:
        g = g / v
    return a + g*(b-a)


def reverse(x):
    """
    Reverse a 1D array or sequence.

    Args:
        x (array-like): Input sequence (e.g., list, NumPy array).

    Returns:
        array-like: Reversed sequence.
    """
    
    return x[::-1]


def normalise(x):
    """
    Normalise an image

    Args:
        x (array-like): Input sequence (e.g., list, NumPy array).

    Returns:
        array-like: normalise between 0 and 1
    """

    return (x - x.min()) / (x.max() - x.min())


def snr(x, y):
    """
    snr - signal to noise ratio

       v = snr(x,y);

     v = 20*log10( norm(x(:)) / norm(x(:)-y(:)) )

       x is the original clean signal (reference).
       y is the denoised signal.

    Copyright (c) 2014 Gabriel Peyre
    """

    return 20 * np.log10(pyl.norm(x) / pyl.norm(x - y))
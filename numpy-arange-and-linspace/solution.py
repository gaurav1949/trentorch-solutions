import numpy as np


def make_range(start: float, stop: float, step: float) -> np.ndarray:
    """
    Return an array of values from `start` up to (but not
    including) `stop`, incrementing by `step`, using np.arange.
    """
    return np.arange(start,stop,step)


def make_evenly_spaced(start: float, stop: float, count: int) -> np.ndarray:
    """
    Return an array of exactly `count` evenly-spaced values from
    `start` to `stop`, inclusive of both endpoints, using
    np.linspace.
    """
    return np.linspace(start,stop,count)


def compare_arange_linspace(start: float, stop: float, step: float) -> dict:
    """
    Using the same start/stop/step values:
      1. Generate an array with np.arange(start, stop, step).
      2. Compute how many elements that array has (its length).
      3. Generate a second array with np.linspace using that
         same length as `num`, over the same start/stop range.
     
    Return a dictionary:
      {
        "arange_result": <the arange array>,
        "linspace_result": <the linspace array>,
        "arange_includes_stop": <True if stop is in arange_result>,
        "linspace_includes_stop": <True if stop is in linspace_result>
      }
    """
    a=np.arange(start,stop,step)
    length=len(a)
    num=len(a)
    b=np.linspace(start,stop,num)
    return {"arange_result":a,"linspace_result":b,"arange_includes_stop":stop in a,"linspace_includes_stop":stop in b}

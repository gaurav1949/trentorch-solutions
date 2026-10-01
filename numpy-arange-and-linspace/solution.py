import numpy as np


def make_range(start: float, stop: float, step: float) -> np.ndarray:
    
    return np.arange(start,stop,step)


def make_evenly_spaced(start: float, stop: float, count: int) -> np.ndarray:
    
    return np.linspace(start,stop,count)


def compare_arange_linspace(start: float, stop: float, step: float) -> dict:
    
    a=np.arange(start,stop,step)
    length=len(a)
    num=len(a)
    b=np.linspace(start,stop,num)
    return {"arange_result":a,"linspace_result":b,"arange_includes_stop":stop in a,"linspace_includes_stop":stop in b}

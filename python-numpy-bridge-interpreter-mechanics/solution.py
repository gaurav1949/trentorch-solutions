import math


def dot_loop(a: list, b: list) -> float:
    """
    Return the dot product of `a` and `b` (two lists of numbers of
    the same length): the sum of a[i] * b[i] over all positions.
    Use a plain for loop with an index (range and len). An empty
    input returns 0.0.
    """
    sum=0;
    for i in range(len(a)):
        sum+=a[i]*b[i]
    return sum


def loop_time_model(n: int, per_iteration_cost: float) -> float:
    """
    Return the modeled time of a Python loop over n elements:
    n * per_iteration_cost. If n <= 0, return 0.0.
    """
    if n<=0:
        return 0.0
    else:
        return float(n * per_iteration_cost)


def vectorized_time_model(n: int, call_overhead: float, per_element_cost: float) -> float:
    """
    Return the modeled time of one vectorized operation over n
    elements: call_overhead + n * per_element_cost. If n <= 0,
    return call_overhead.
    """
    if n<=0:
        return call_overhead
    return call_overhead + n * per_element_cost


def break_even_n(per_iteration_cost: float, call_overhead: float, per_element_cost: float):
    """
    Return the smallest integer n >= 1 for which the vectorized
    time is less than or equal to the loop time, i.e.
        call_overhead + n * per_element_cost <= n * per_iteration_cost
    which is n >= call_overhead / (per_iteration_cost - per_element_cost).
    If per_element_cost >= per_iteration_cost, vectorization never
    wins for large n: return None. Use math.ceil, and never
    return a value smaller than 1.
    """
    if per_element_cost >= per_iteration_cost:
        return None
    result=call_overhead / (per_iteration_cost - per_element_cost)
    return max(1,math.ceil(result))
    
def list_of_ints_bytes(n: int) -> int:
    """
    Estimate the bytes used by a list of n distinct small ints:
    each element needs an 8-byte slot plus a 28-byte int object.
    Ignore the list's own header. Return n * 36 (0 if n <= 0).
    """
    if n<=0:
        return 0
    return n*36


def typed_array_bytes(n: int, itemsize: int) -> int:
    """
    Return the bytes used by a typed array of n elements that are
    each `itemsize` bytes: n * itemsize (0 if n <= 0).
    """
    if n<=0:
        return 0
    return n*itemsize

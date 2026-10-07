import numpy as np


def missing_mask(x: np.ndarray) -> np.ndarray:
    """
    Missing numeric values are represented as np.nan (NumPy/pandas'
    standard convention). Returns a boolean array the same shape as x,
    True wherever a value is missing.

    NOTE: `x == np.nan` is ALWAYS False, even for an actual NaN (NaN is
    defined to never equal anything, including itself), np.isnan is
    the correct tool here, not ==.
    """
    return np.isnan(x)


def missing_count_per_column(x: np.ndarray) -> np.ndarray:
    """
    x is a 2D array, (rows, columns). Returns a length-`num_columns`
    array: how many missing values each column has.
    """
    return np.sum(missing_mask(x),axis=0)


def missing_fraction_per_column(x: np.ndarray) -> np.ndarray:
    """
    Same as missing_count_per_column, but as a fraction of the total
    row count (0.0 to 1.0) rather than a raw count.
    """
    num_rows=x.shape[0]
    result=missing_count_per_column(x)/num_rows
    return result


def impute_with_mean(x: np.ndarray) -> np.ndarray:
    """
    Replace every missing (NaN) value with its OWN COLUMN's mean,
    computed from that column's non-missing values only
    (np.nanmean ignores NaNs automatically, rather than propagating
    them into the mean itself).

    Must not mutate the input array, work on a copy.
    """
    result=x.copy()
    cols_mean=np.nanmean(result,axis=0)
    inds=np.where(np.isnan(result))
    result[inds]=np.take(cols_mean,inds[1])
    return result

def impute_with_median(x: np.ndarray) -> np.ndarray:
    """
    Same idea as impute_with_mean, but replacing with each column's
    median (np.nanmedian) instead. See Theory for why median is
    sometimes the better choice.
    """
    result=x.copy()
    cols_median=np.nanmedian(result,axis=0)
    inds=np.where(np.isnan(result))
    result[inds]=np.take(cols_median,inds[1])
    return result

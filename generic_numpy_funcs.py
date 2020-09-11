def partial_vectorization_fit(func, iterable, *params):
    """
    Purpose: Given an iterable, initialize a partial vectorized function with given parameters and then iterate through
             and return results in the form of a ndarray.

    :param func: A pickled function that you want vectorized.
    :param iterable: The iterable to initialize the function with.
    :param params: List of partial parameters to initialize func with. Must be k-1 params, where k = num params.
    :return: ndarray of results of the function.
    """
    partial_func = np.vectorize(partial(func, *params))
    results = partial_func(iterable)
    return results


def sort_by_position(data, field_idx):
    """
    Purpose : Takes an ndarray with a column index and sorts on that column, returning a data object with a new dim that
              represents the original index positions of the pre-sorted objects.

    :param data: ndarray
    :param field_idx: index position of a column within the ndarray. If an NxM array, field_idx <= M
    :return: sorted ndarray with a new field representing the sorted rows original index.
    """
    ordered_keys = np.argsort(data[:, field_idx])
    ordered_strata_scores = np.concatenate([data[ordered_keys], ordered_keys[:, None]], axis=1)
    return ordered_strata_scores

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras


def _normalize_dtype(value):
    if isinstance(value, tf.dtypes.DType):
        return value.as_numpy_dtype
    if hasattr(value, "name") and isinstance(value.name, str):
        return value.name
    return value


def _wrap_numpy_with_dtype(fn, dtype_index=None):
    def wrapper(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = _normalize_dtype(kwargs["dtype"])
        elif dtype_index is not None and len(args) > dtype_index:
            args = list(args)
            args[dtype_index] = _normalize_dtype(args[dtype_index])
            args = tuple(args)
        return fn(*args, **kwargs)

    return wrapper


@pytest.fixture(autouse=True, scope="session")
def _normalize_keras_floatx():
    original_floatx = keras.backend.floatx

    def normalized_floatx():
        value = original_floatx()
        return value.name if hasattr(value, "name") else value

    keras.backend.floatx = normalized_floatx
    original_numpy = {
        "array": np.array,
        "zeros": np.zeros,
        "ones": np.ones,
        "empty": np.empty,
        "full": np.full,
    }
    np.array = _wrap_numpy_with_dtype(np.array, dtype_index=1)
    np.zeros = _wrap_numpy_with_dtype(np.zeros, dtype_index=1)
    np.ones = _wrap_numpy_with_dtype(np.ones, dtype_index=1)
    np.empty = _wrap_numpy_with_dtype(np.empty, dtype_index=1)
    np.full = _wrap_numpy_with_dtype(np.full, dtype_index=2)
    yield
    np.array = original_numpy["array"]
    np.zeros = original_numpy["zeros"]
    np.ones = original_numpy["ones"]
    np.empty = original_numpy["empty"]
    np.full = original_numpy["full"]
    keras.backend.floatx = original_floatx

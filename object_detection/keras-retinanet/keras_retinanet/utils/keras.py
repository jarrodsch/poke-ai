from tensorflow import keras
import tensorflow as tf


def keras_floatx():
    value = keras.backend.floatx()
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, tf.dtypes.DType):
        return value.as_numpy_dtype
    return value

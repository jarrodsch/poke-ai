from tensorflow import keras as _keras

_original_floatx = _keras.backend.floatx


def _normalized_floatx():
    value = _original_floatx()
    return value.name if hasattr(value, "name") else value


_keras.backend.floatx = _normalized_floatx

import pytest
from tensorflow import keras


@pytest.fixture(autouse=True, scope="session")
def _normalize_keras_floatx():
    original_floatx = keras.backend.floatx

    def normalized_floatx():
        value = original_floatx()
        return value.name if hasattr(value, "name") else value

    keras.backend.floatx = normalized_floatx
    yield
    keras.backend.floatx = original_floatx

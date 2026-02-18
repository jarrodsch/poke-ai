import tensorflow as tf


def configure_tensorflow():
    """Configure TensorFlow for modern TF2.x behavior and safer GPU memory use."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            # Allow TF to grow GPU memory usage as needed.
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # If configuration is unsupported or already set, ignore.
        pass

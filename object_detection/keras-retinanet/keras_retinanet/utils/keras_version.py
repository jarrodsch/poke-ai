"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

import re

import tensorflow as tf
from tensorflow import keras
import sys

minimum_keras_version = 2, 10, 0


def _parse_version(value):
    parts = []
    for item in value.split('.'):
        match = re.search(r'\d+', item)
        if match:
            parts.append(int(match.group(0)))
        else:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _get_keras_version():
    version = getattr(keras, '__version__', None)
    if version:
        return version
    try:
        import keras as standalone_keras
        version = getattr(standalone_keras, '__version__', None)
        if version:
            return version
    except Exception:
        pass
    return tf.__version__


def keras_version():
    """ Get the Keras version.

    Returns
        tuple of (major, minor, patch).
    """
    return _parse_version(_get_keras_version())


def keras_version_ok():
    """ Check if the current Keras version is higher than the minimum version.
    """
    return keras_version() >= minimum_keras_version


def assert_keras_version():
    """ Assert that the Keras version is up to date.
    """
    detected = _get_keras_version()
    required = '.'.join(map(str, minimum_keras_version))
    assert(keras_version() >= minimum_keras_version), 'You are using keras version {}. The minimum required version is {}.'.format(detected, required)


def check_keras_version():
    """ Check that the Keras version is up to date. If it isn't, print an error message and exit the script.
    """
    try:
        assert_keras_version()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

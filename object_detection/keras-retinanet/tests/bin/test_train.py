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

import os
import warnings

import pytest
from tensorflow import keras

import keras_retinanet.bin.train

TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test-data'))


@pytest.fixture(autouse=True)
def clear_session():
    # run before test (do nothing)
    yield
    # run after test, clear keras session
    keras.backend.clear_session()


def test_coco():
    try:
        import pycocotools  # noqa: F401
    except Exception:
        pytest.skip('pycocotools is not installed')

    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        'coco',
        os.path.join(TEST_DATA_DIR, 'coco'),
    ])


def test_pascal():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        'pascal',
        os.path.join(TEST_DATA_DIR, 'pascal'),
    ])


def test_csv():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        'csv',
        os.path.join(TEST_DATA_DIR, 'csv', 'annotations.csv'),
        os.path.join(TEST_DATA_DIR, 'csv', 'classes.csv'),
    ])


def test_vgg():
    try:
        import pycocotools  # noqa: F401
    except Exception:
        pytest.skip('pycocotools is not installed')

    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--backbone=vgg16',
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--no-snapshots',
        '--freeze-backbone',
        'coco',
        os.path.join(TEST_DATA_DIR, 'coco'),
    ])

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

from tensorflow import keras
from ..utils.coco_eval import evaluate_coco


def _get_tensorboard_writer(tensorboard):
    if tensorboard is None:
        return None
    writer = getattr(tensorboard, "writer", None)
    if writer is not None:
        return writer
    writers = getattr(tensorboard, "_writers", None)
    if isinstance(writers, dict) and writers:
        return writers.get("train") or next(iter(writers.values()))
    get_writer = getattr(tensorboard, "_get_writer", None)
    if callable(get_writer):
        try:
            return get_writer("train")
        except Exception:
            return None
    return None


class CocoEval(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """
    def __init__(self, generator, tensorboard=None, threshold=0.05):
        """ CocoEval callback intializer.

        Args
            generator   : The generator used for creating validation data.
            tensorboard : If given, the results will be written to tensorboard.
            threshold   : The score threshold to use.
        """
        self.generator = generator
        self.threshold = threshold
        self.tensorboard = tensorboard

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']
        coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold)
        writer = _get_tensorboard_writer(self.tensorboard)
        if coco_eval_stats is not None and writer is not None:
            import tensorflow as tf
            if hasattr(writer, "add_summary"):
                summary = tf.compat.v1.Summary()
                for index, result in enumerate(coco_eval_stats):
                    summary_value = summary.value.add()
                    summary_value.simple_value = result
                    summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                    writer.add_summary(summary, epoch)
                    logs[coco_tag[index]] = result
            else:
                with writer.as_default():
                    for index, result in enumerate(coco_eval_stats):
                        tag = '{}. {}'.format(index + 1, coco_tag[index])
                        tf.summary.scalar(tag, result, step=epoch)
                        logs[coco_tag[index]] = result
                if hasattr(writer, "flush"):
                    writer.flush()

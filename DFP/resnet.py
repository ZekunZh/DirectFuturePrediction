import numpy as np
import tensorflow as tf
import tempfile
import logging

from keras.optimizers import SGD
from keras import backend as K

RGB_MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3)).astype(np.float32)
DEFAULT_IMAGE_SHAPE = (1, 224, 224, 3)


class ResNet50():
    """
    A class that builds a TF graph with a pre-trained ResNet50 model (on imagenet)
    Also takes care of preprocessing. Input should be a regular RGB image (0-255)
    """

    def __init__(self, image_shape=DEFAULT_IMAGE_SHAPE, input_tensor=None):
        self.image_shape = image_shape
        self._build_graph(input_tensor)

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope('ResNet'):
                with tf.name_scope('inputs'):
                    if input_tensor is None:
                        input_tensor = tf.placeholder(tf.float32, shape=self.image_shape, name='input_img')
                    else:
                        assert self.image_shape == input_tensor.shape
                    self.input_tensor = input_tensor

                with tf.name_scope('preprocessing'):
                    # img = self.input_tensor - RGB_MEAN_PIXELS
                    img = self.input_tensor
                    #img = tf.reverse(img, axis=[-1])
                    logging.info("input images shape: " + str(img.get_shape()))

                with tf.variable_scope('model'):
                    self.resnet = tf.keras.applications.ResNet50(weights='imagenet',
                                                             include_top=True, input_tensor=img)
                    logging.info(self.resnet.summary())
                    # for layer in self.resnet.layers:
                    #     layer.trainable = False

                    # self.resnet.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

                self.outputs = {l.name: l.output for l in self.resnet.layers}

            self.resnet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ResNet/model')
            logging.info("trainable weights of ResNet: " + str(self.resnet_weights))

            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(self.resnet_weights).save(sess, f.name)

        self.model_weights_tensors = set(self.resnet_weights)

    def load_weights(self, session):
        # sess = tf.get_default_session()
        sess = session
        tf.train.Saver(self.resnet_weights).restore(sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.outputs[key]
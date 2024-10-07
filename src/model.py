from tensorflow import keras
import tensorflow as tf
from layers import *

class UNet(keras.Model):
    def __init__(self, input_size, n_filters, n_classes):
        input_tensor = layers.Input(input_size)

        conv, pool = [], []

        down_depth = 4
        for i in range(down_depth):
            x = input_tensor if not pool else pool[-1]
            
            conv_i = double_conv(x, n_filters * 2 ** i)
            pool_i = pooling(conv_i)
            
            conv.append(conv_i)
            pool.append(pool_i)

        last_conv = double_conv(pool[-1], n_filters * 2 ** (down_depth))

        x = last_conv
        up_depth = 4
        for i in range(up_depth):
            x = deconvolution(x, n_filters * 2 ** (up_depth - i - 1))
            x = join_skip(x, conv[-i-1])
            x = double_conv(x, n_filters * 2 ** (up_depth - i - 1))

        output_tensor = single_output_conv(x, n_filters=n_classes)

        super(UNet, self).__init__(input_tensor, output_tensor, name="UNet")

    def train_step(self, data):
        x, y, mask = data

        # Forward pass
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss(
                y=y,
                y_pred=y_pred,
                sample_weight=mask,
            )
            self._loss_tracker.update_state(
                loss
            )
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        return self.compute_metrics(x, y, y_pred)

    def test_step(self, data):
        x, y, mask = data
        y_pred = self(x, training=False)
        loss = self.loss(
            y=y, y_pred=y_pred, 
            sample_weight=mask, 
        )
        self._loss_tracker.update_state(
            loss
        )
        return self.compute_metrics(x, y, y_pred)

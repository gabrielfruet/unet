def weight_map_focal_loss(y, y_pred, sample_weight, gamma=2.0, alpha=0.25):
    """
    Custom loss function using binary focal loss with a mask.
    
    y_true: ground truth labels
    y_pred: predicted labels
    mask: a mask with shape [BATCH_SIZE, HEIGHT, WIDTH] to weight the loss
    gamma: focusing parameter for focal loss (default=2.0)
    alpha: balancing factor for focal loss (default=0.25)
    """
    mask = tf.cast(sample_weight, dtype=tf.float32)  # Convert mask to float32 if necessary

    loss = keras.losses.binary_focal_crossentropy(y, y_pred)
    
    loss *= mask

    return tf.reduce_mean(loss)

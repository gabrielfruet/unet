from tensorflow.keras import layers, ops

def double_conv(input_tensor, n_filters, kernel_size=3):
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                      padding='same', kernel_initializer = 'he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                      padding='same', kernel_initializer = 'he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def pooling(input_tensor, dropout_rate = 0.1):
    x = layers.MaxPool2D(pool_size=(2, 2))(input_tensor)
    x = layers.Dropout(rate = dropout_rate)(x)
    return x

def single_output_conv(input_tensor, kernel_size=1, n_filters=2):
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                      padding='same', kernel_initializer = 'he_normal')(input_tensor)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)
    #if n_filters == 1:
        #x = layers.Reshape((input_tensor.shape[1], input_tensor.shape[2]))(x)
    return x

def deconvolution(input_tensor, n_filters, kernel_size=3, stride=2):
    x = layers.Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                               strides = (stride, stride), padding='same', kernel_initializer = 'he_normal')(input_tensor)
    return x

def join_skip(input_tensor, skip):
    return layers.concatenate([skip, input_tensor])

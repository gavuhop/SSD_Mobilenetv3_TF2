import tensorflow as tf
import numpy as np
# x = np.random.randn(10, 15, 15, 20) # Batch_size, W, H, C
# x = x.astype('float32')
def depthwise_separable_conv_tf(x):
  dw2d = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3, strides=1, padding='same', depth_multiplier=1
  )
  pw2d = tf.keras.layers.Conv2D(
      filters=50, kernel_size=1, strides=1
  )
  y = dw2d(x)
  y = pw2d(y)
  return y
# y = depthwise_separable_conv_tf(x)
# aaa = y.shape # Batch_size, W, H, C
# print(aaa)

x = np.random.randn(10, 64, 64, 16)
x = x.astype('float32')

def inverted_linear_residual_block(x, expand=64, squeeze=16):
  '''
  expand: số lượng channel của layer trung gian
  squeeze: số lượng channel của layer bottleneck input và output
  '''
  # Depthwise convolution
  m = tf.keras.layers.Conv2D(expand, (1,1), padding='SAME', activation='relu')(x)
  m = tf.keras.layers.DepthwiseConv2D((3,3), padding='SAME', activation='relu')(m)
  # Pointwise convolution + Linear projection
  m = tf.keras.layers.Conv2D(squeeze, (1,1), padding='SAME', activation='linear')(m)
  opt = tf.keras.layers.Add()([m, x])
  return opt

y = inverted_linear_residual_block(x, expand=64, squeeze=16)
y.shape
print(y.shape)
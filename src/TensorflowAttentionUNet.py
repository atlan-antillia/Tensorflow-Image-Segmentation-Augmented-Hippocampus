# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This is based on the code in the following web sites:

# 1. Semantic-Segmentation-Architecture
# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/attention-unet.py

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train_eval_infer.config


import os
import sys
import traceback

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import (Conv2D, Dropout, Conv2D, MaxPool2D, 
                                     Activation, BatchNormalization, UpSampling2D, Concatenate)

from tensorflow.keras import Model
from ConfigParser import ConfigParser
from TensorflowUNet import TensorflowUNet


class TensorflowAttentionUNet(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)
  
  # The following methods have been take from the following code.
  # https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/attention-unet.py

  def conv_block(self, x, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    # 2024/03/29
    if self.dropout_rate>0.0:
      print("--- inserted Dropout")
      x = Dropout(self.dropout_rate)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

  def encoder_block(self, x, num_filters):
    x = self.conv_block(x, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

  def attention_gate(self, g, s, num_filters):
    Wg = Conv2D(num_filters, 1, padding="same")(g)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(num_filters, 1, padding="same")(s)
    # 2024/03/29
    if self.dropout_rate>0.0:
      print("--- inserted Dropout")
      Ws = Dropout(self.dropout_rate)(Ws)

    Ws = BatchNormalization()(Ws)

    out = Activation("relu")(Wg + Ws)
    out = Conv2D(num_filters, 1, padding="same")(out)
    out = Activation("sigmoid")(out)

    return out * s

  def decoder_block(self, x, s, num_filters):
    x = UpSampling2D(interpolation="bilinear")(x)
    s = self.attention_gate(x, s, num_filters)
    x = Concatenate()([x, s])
    x = self.conv_block(x, num_filters)
    return x

  # Customizable by the parameters in a configuration file.
  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("=== TensorflowAttentionUNet.create ")
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    self.dropout_rate = self.config.get(ConfigParser.MODEL, "dropout_rate", dvalue=0.1)
    print("=== dropout_rate {}".format(self.dropout_rate))

    inputs = Input((image_height, image_width, image_channels))

    #inputs = Input((image_height, image_width, image_channels))
    # 2024/03/31 commentted out the following line.
    #p = Lambda(lambda x: x / 255)(inputs)
    p = inputs
    enc = []
    d   = None
    for i in range(num_layers):
      filters = base_filters * (2**i)
      print("--- encoder filters {}".format(filters))
      if i < num_layers-1:
        s, p    = self.encoder_block(p, filters)
        enc.append(s)
      else:
        d = self.conv_block(p, filters)
        
    enc_len = len(enc)
    enc.reverse()
    n = 0
    for i in range(num_layers-1):
      f = enc_len - 1 - i
      filters = base_filters* (2**f)
      print("--- decoder filters {}".format(filters))

      s = enc[n]
      d = self.decoder_block(d, s, filters)
      n += 1

    # 2024/03/29 Added the following line.
    activation = "softmax"
    if num_classes == 1:
      activation = "sigmoid"
    outputs = Conv2D(num_classes, (1, 1), activation=activation)(d)

    model = Model(inputs=[inputs], outputs=[outputs], name="Attention-UNET")

    return model

    
if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    print("=== config_file {}".format(config_file))

    config   = ConfigParser(config_file)

    width    = config.get(ConfigParser.MODEL, "image_width")
    height   = config.get(ConfigParser.MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowAttentionUNet(config_file)
    
  except:
    traceback.print_exc()
    

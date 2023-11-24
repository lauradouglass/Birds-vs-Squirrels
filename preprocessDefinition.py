{\rtf1\ansi\ansicpg1252\cocoartf2757
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww28600\viewh14860\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #necessary libraries\
import tensorflow as tf\
from tensorflow import keras\
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2\
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception\
\
#preprocess function depending on the model\
def prep(image, label, model_name):\
    if model_name == 'xception':\
        processed_image = tf.image.resize_with_pad(image, 299, 299)\
        out_image = preprocess_xception(processed_image)\
    elif model_name == 'mobilenetv2':\
        processed_image = tf.image.resize(tf.image.convert_image_dtype(image, tf.float32), (224, 224))\
        out_image = preprocess_mobilenetv2(processed_image)\
    else:\
        raise ValueError(\'93Not ild27s preprocessor\'94)\
    \
    return out_image, label\
}
{\rtf1\ansi\ansicpg1252\cocoartf2757
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 !jupyter nbconvert --to script preprocessDefinition.ipynb #nb to py\
import tensorflow as tf\
from tensorflow import keras\
from tensorflow.keras.applications import MobileNetV2\
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\
from preprocessDefinition import prep\
\
#new \
model_name_birds_vs_squirrels = 'mobilenetv2'\
\
train_tfrecord_path = 'Downloads/birds-vs-squirrels-train.tfrecords'\
validation_tfrecord_path = 'Downloads/birds-vs-squirrels-validation.tfrecords'\
\
#load model\
pretrained_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\
\
for layer in pretrained_model.layers:\
    layer.trainable = False\
\
#build model\
model = keras.Sequential([\
    pretrained_model,\
    keras.layers.GlobalAveragePooling2D(),\
    keras.layers.Dense(512, activation='relu'),\
    keras.layers.BatchNormalization(),\
    keras.layers.Dropout(0.3),\
    keras.layers.Dense(256, activation='relu'),\
    keras.layers.BatchNormalization(),\
    keras.layers.Dropout(0.3),\
    keras.layers.Dense(3, activation='softmax')\
])\
\
model.compile(optimizer='adam',\
              loss='sparse_categorical_crossentropy',\
              metrics=['accuracy'])\
\
\
#function to parse\
def parse_examples(serialized_examples):\
    feature_description = \{\
        'image': tf.io.FixedLenFeature([], tf.string),\
        'label': tf.io.FixedLenFeature([], tf.int64)\
    \}\
    examples = tf.io.parse_example(serialized_examples, feature_description)\
    targets = examples['label']\
    images = tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'], channels=3), tf.float32), 224, 224)\
    return images, targets\
\
#new\
#training dataset\
raw_train_dataset = tf.data.TFRecordDataset([train_tfrecord_path])\
train_dataset = raw_train_dataset.map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
train_dataset = train_dataset.map(lambda image, label: prep(image, label, model_name_birds_vs_squirrels), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
train_dataset = train_dataset.cache().shuffle(buffer_size=1000).batch(64).prefetch(tf.data.experimental.AUTOTUNE)\
\
#validation dataset\
raw_validation_dataset = tf.data.TFRecordDataset([validation_tfrecord_path])\
validation_dataset = raw_validation_dataset.map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
validation_dataset = validation_dataset.map(lambda image, label: prep(image, label, model_name_birds_vs_squirrels), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
validation_dataset = validation_dataset.cache().batch(64).prefetch(tf.data.experimental.AUTOTUNE)\
\
#learning rate\
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(\
    initial_learning_rate=1e-5,\
    decay_steps=10000,\
    decay_rate=0.9\
)\
\
#use learning rate scheduler on model\
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_scheduler),loss='sparse_categorical_crossentropy', metrics=['accuracy'])\
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)\
model.fit(train_dataset, epochs=20, validation_data=validation_dataset, callbacks=[early_stopping])\
model.save('birds_vs_squirrels_mobilenet_model.keras')}
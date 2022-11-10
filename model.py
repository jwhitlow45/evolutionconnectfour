import tensorflow as tf
import keras.layers as tfl

from board import BOARD_WIDTH, BOARD_HEIGHT

def create_model():
    leaky_relu = lambda x: max(x, 0.3)
    
    model = tf.keras.Sequential()
    model.add(tfl.Input((BOARD_HEIGHT, BOARD_WIDTH, 1)))
    model.add(tfl.Conv2D(filters=64, kernel_size=5, padding='same'))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same'))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same'))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same'))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(64, activation=leaky_relu))
    model.add(tfl.Dense(64, activation=leaky_relu))
    model.add(tfl.Dense(64, activation=leaky_relu))
    model.add(tfl.Dense(BOARD_WIDTH, activation='sigmoid'))
    
    model.compile(optimizer='adam')
    
    return model
import tensorflow as tf
import keras.layers as tfl

from board import BOARD_WIDTH, BOARD_HEIGHT

def create_model():
    
    model = tf.keras.Sequential()
    model.add(tfl.Input((BOARD_HEIGHT, BOARD_WIDTH, 1)))
    model.add(tfl.Conv2D(filters=64, kernel_size=5, padding='same'))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same'))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same'))
    model.add(tfl.Conv2D(filters=64, kernel_size=4, padding='same'))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(64))
    model.add(tfl.Dense(64))
    model.add(tfl.Dense(64))
    model.add(tfl.Dense(BOARD_WIDTH))
    
    model.compile(optimizer='adam')
    
    return model
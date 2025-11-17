import tensorflow as tf
from .callbacks_custom import ASK

def create_callbacks(model, epochs, ask_epoch):
    ask = ASK(model, epochs, ask_epoch)
    rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, verbose=1, restore_best_weights=True)
    callbacks = [rlronp, estop, ask]
    return callbacks
def train_model(model, train_gen, valid_gen, epochs, callbacks):
    history = model.fit(x=train_gen, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=valid_gen,
                       validation_steps=None, shuffle=False, initial_epoch=0)
    return history
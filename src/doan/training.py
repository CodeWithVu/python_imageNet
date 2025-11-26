def train_model(model, train_gen, valid_gen, epochs, callbacks):
    history = model.fit(x=train_gen, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=valid_gen,
                       validation_steps=None, shuffle=False, initial_epoch=0)
    return history

# x=train_gen
# → dữ liệu train có augmentation.

# epochs=epochs
# → số vòng lặp huấn luyện tối đa (ví dụ 20, 30,…).

# callbacks=callbacks
# → nơi diễn ra phép màu giúp mô hình học tốt hơn.

# validation_data=valid_gen
# → kiểm tra mô hình mỗi epoch.

# shuffle=False
# → giữ nguyên thứ tự ảnh khi lấy từ generator.
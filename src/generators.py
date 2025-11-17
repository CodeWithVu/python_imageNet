from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_df, valid_df, test_df, img_size=(200,200), batch_size=30):
    trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2, height_shift_range=.2, zoom_range=.2)
    t_and_v_gen = ImageDataGenerator()
    
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                         class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    
    length = len(test_df)
    test_batch_size = sorted([int(length/n) for n in range(1, length+1) if length % n == 0 and length/n <= 80], reverse=True)[0]
    test_steps = int(length/test_batch_size)
    test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                              class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
    
    classes = list(train_gen.class_indices.keys())
    class_count = len(classes)
    print('test batch size: ', test_batch_size, ' test steps: ', test_steps, ' number of classes: ', class_count)
    return train_gen, valid_gen, test_gen, classes, class_count, test_steps
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax

def create_model(img_shape, class_count, lr=0.001):
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
    base_model.trainable = True # fine tuning full network
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(
        256,
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006),
        activation='relu'
    )(x)
    x = Dropout(rate=.4, seed=123)(x)
    output = Dense(class_count, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# include_top=False
# → Bỏ phần classification gốc của EfficientNet (phần Dense 1000 lớp của ImageNet).
# Vì bạn dùng dataset AID nên số lớp khác.

# weights="imagenet"
# → Load trọng số pretrained.

# input_shape=img_shape
# → Dữ liệu đầu vào của bạn (ví dụ 224×224×3).

# pooling='max'
# → Lấy Global Max Pooling để chuyển feature map 2D → vector 1D.
# Điều này giúp:

# giảm số tham số

# ngăn overfitting

# dễ dùng Dense layer ở phía sau



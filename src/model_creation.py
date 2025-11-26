import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax

def create_model(img_shape, class_count, lr=0.0008):
    """
    Tạo model EfficientNetB3 - Tương thích với Keras 3.x (TensorFlow 2.16+)
    
    Các thay đổi quan trọng cho Keras 3.x:
    1. Sử dụng Functional API với tf.keras.Input()
    2. Thêm preprocessing layer cho EfficientNet
    3. Giảm regularization (từ 0.016 xuống 0.001)
    4. Giảm learning rate (từ 0.001 xuống 0.0008)
    """
    # Load base model
    base_model = tf.keras.applications.VGG16(
        include_top=False, 
        weights="imagenet", 
        input_shape=img_shape, 
        pooling='max'
    )
    base_model.trainable = True  # Fine tuning full network
    
    # Build model với Functional API (quan trọng cho Keras 3.x)
    inputs = tf.keras.Input(shape=img_shape)
    
    # QUAN TRỌNG: Thêm preprocessing layer cho VGG16
    # ImageDataGenerator output [0-255], VGG16 cần preprocessing đặc biệt
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    
    x = base_model(x, training=True)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    
    # GIẢM REGULARIZATION cho Keras 3.x (tránh accuracy bị chặn)
    x = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    x = Dropout(rate=0.45, seed=123)(x)
    output = Dense(class_count, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"✅ Model created for Keras 3.x (TF {tf.__version__})")
    print(f"   - L2 regularization: 0.001")
    print(f"   - Learning rate: {lr}")
    
    return model

# include_top=False
# → Bỏ phần classification gốc của VGG16 (phần Dense 1000 lớp của ImageNet).
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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax

def create_model(img_shape, class_count, lr=0.0008, debug: bool = False):
    """
    Tạo model EfficientNetB3 - Tương thích với Keras 3.x (TensorFlow 2.16+)
    
    Các thay đổi quan trọng cho Keras 3.x:
    1. Sử dụng Functional API với tf.keras.Input()
    2. Thêm preprocessing layer cho EfficientNet
    3. Giảm regularization (từ 0.016 xuống 0.001)
    4. Giảm learning rate (từ 0.001 xuống 0.0008)
    """
    # Load base model
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, 
        weights="imagenet", 
        input_shape=img_shape, 
        pooling='max' #vector lúc này  là (7*7*1536) -> (batch_size, 1536)
    )
    base_model.trainable = True  # Fine tuning full network
    
    # Build model với Functional API (quan trọng cho Keras 3.x)
    inputs = tf.keras.Input(shape=img_shape)
    
    # QUAN TRỌNG: Thêm preprocessing layer cho EfficientNet
    # ImageDataGenerator output [0-255], EfficientNet cần preprocessing đặc biệt
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    if debug:
        print("[DEBUG] After preprocess_input ->", type(x), "shape:", x.shape, "dtype:", x.dtype)
    
    x = base_model(x, training=True)
    if debug:
        print("[DEBUG] After base_model ->", type(x), "shape:", x.shape, "dtype:", x.dtype)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    if debug:
        print("[DEBUG] After BatchNormalization ->", type(x), "shape:", x.shape, "dtype:", x.dtype)
    
    # GIẢM REGULARIZATION cho Keras 3.x (tránh accuracy bị chặn)
    x = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)  # ReLU(z) = max(0, z)
    if debug:
        print("[DEBUG] After Dense(256) ->", type(x), "shape:", x.shape, "dtype:", x.dtype)
    x = Dropout(rate=0.45, seed=123)(x)
    if debug:
        print("[DEBUG] After Dropout ->", type(x), "shape:", x.shape, "dtype:", x.dtype)
    output = Dense(class_count, activation='softmax')(x)  #   softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
    if debug:
        print("[DEBUG] Final outputs ->", type(output), "shape:", output.shape, "dtype:", output.dtype)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    if debug:
        try:
            model.summary()
        except Exception:
            pass
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"✅ Model created for Keras 3.x (TF {tf.__version__})")
    print(f"   - L2 regularization: 0.001")
    print(f"   - Learning rate: {lr}")
    
    return model

# include_top=False
# → Bỏ phần classification gốc của EfficientNet (phần Dense 1000 lớp của ImageNet).


# weights="imagenet"
# → Load trọng số pretrained.

# input_shape=img_shape
# → Dữ liệu đầu vào 

# pooling='max'
# → Lấy Global Max Pooling để chuyển feature map 2D → vector 1D.
# Điều này giúp:

# giảm số tham số

# ngăn overfitting

# dễ dùng Dense layer ở phía sau



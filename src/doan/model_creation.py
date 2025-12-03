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
    2. Thêm preprocessing layer cho EfficientNetB3
    3. Giảm regularization (từ 0.016 xuống 0.001)
    4. Giảm learning rate (từ 0.001 xuống 0.0008)
    """
    # Load base model
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False, # Bỏ phần classification gốc của EfficientNet
        weights="imagenet", # Load trọng số pretrained của ImageNet
        input_shape=img_shape, # Dữ liệu đầu vào của tôi
        pooling='max' # Lấy Global Max Pooling để chuyển feature map 2D → vector 1D
    )
    base_model.trainable = True  # Fine tuning full network
    
    # Build model với Functional API (quan trọng cho Keras 3.x)
    inputs = tf.keras.Input(shape=img_shape)
    
    # QUAN TRỌNG: Thêm preprocessing layer cho EfficientNetB3
    # ImageDataGenerator output [0-255], EfficientNetB3 cần preprocessing đặc biệt -> [-1, 1]
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    x = base_model(x, training=True) # Ensure base_model is in training mode for BatchNorm layers
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    # axis=-1 chuẩn hóa theo channel cuối vì dữ liệu hình ảnh có định dạng (batch_size, height, width, channels)
    
    # GIẢM REGULARIZATION cho Keras 3.x (tránh accuracy bị chặn)
    # ReLU (Rectified Linear Unit) activation function
    # Formula: f(x) = max(0, x)
    # It outputs the input directly if it is positive, otherwise, it will output zero.
    x = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    x = Dropout(rate=0.45, seed=123)(x)

     # Softmax activation function
    # Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in range(num_classes))
    # It converts a vector of numbers into a probability distribution, where each value is between 0 and 1, and the sum of all values is 1.
    output = Dense(class_count, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    # loss='categorical_crossentropy' vì dùng one-hot encoding
    
    print(f"✅ EfficientNetB3 Model created for Keras 3.x (TF {tf.__version__})")
    print(f"   - L2 regularization: 0.001")
    print(f"   - Learning rate: {lr}")
    
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



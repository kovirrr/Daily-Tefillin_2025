from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate, Add
from tensorflow.keras import backend as K

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def residual_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    res = Conv2D(num_filters, 1, padding='same')(input_tensor)
    res = BatchNormalization()(res)
    x = Add()([x, res])
    return x

def unet_resnet(input_size=(512, 512, 3)):
    inputs = Input(input_size)
    
    # Encoder
    e1 = residual_block(inputs, 64)
    p1 = MaxPooling2D()(e1)
    
    e2 = residual_block(p1, 128)
    p2 = MaxPooling2D()(e2)
    
    e3 = residual_block(p2, 256)
    p3 = MaxPooling2D()(e3)
    
    e4 = residual_block(p3, 512)
    p4 = MaxPooling2D()(e4)
    
    # Bridge
    b = residual_block(p4, 1024)
    
    # Decoder
    u1 = UpSampling2D()(b)
    u1 = concatenate([u1, e4])
    d1 = residual_block(u1, 512)
    
    u2 = UpSampling2D()(d1)
    u2 = concatenate([u2, e3])
    d2 = residual_block(u2, 256)
    
    u3 = UpSampling2D()(d2)
    u3 = concatenate([u3, e2])
    d3 = residual_block(u3, 128)
    
    u4 = UpSampling2D()(d3)
    u4 = concatenate([u4, e1])
    d4 = residual_block(u4, 64)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(d4)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * K.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, [1,2,3]) + K.sum(y_pred, [1,2,3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
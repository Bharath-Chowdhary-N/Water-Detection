
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Dropout,MaxPooling2D,Conv2DTranspose,Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import losses
from skimage.transform import resize
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

class landsat():
    def __init__(self):
        self.batch_size = 32

        self.img_height = 128
        self.img_width = 128
        # self.load_data()
        self.num_filters = 16
        self.dropout = 0.1
        #self.input_img = mpimg.imread('FinalData/train_gt/image_576.png')
        self.num = 400

        #self.load_data()
        # self.unet_model()





    def compile_and_fit_model(self):
        self.train_generator = zip(image_generator, mask_generator)
        self.model.fit(
            train_generator,
            steps_per_epoch=2000,
            epochs=50)


    def unet_model(self):
        a1 = self.conv_block(self.input_image, num_filter=self.num_filters, kernel_size=3, batch_normalization=True)
        p1 = MaxPooling2D((2, 2))(a1)
        p1 = Dropout(self.dropout)(p1)

        a2 = self.conv_block(p1, num_filter=self.num_filters * 2, kernel_size=3, batch_normalization=True)
        p2 = MaxPooling2D((2, 2))(a2)
        p2 = Dropout(self.dropout)(p2)

        a3 = self.conv_block(p2, num_filter=self.num_filters * 4, kernel_size=3, batch_normalization=True)
        p3 = MaxPooling2D((2, 2))(a3)
        p3 = Dropout(self.dropout)(p3)

        a4 = self.conv_block(p3, num_filter=self.num_filters * 8, kernel_size=3, batch_normalization=True)
        p4 = MaxPooling2D((2, 2))(a4)
        p4 = Dropout(self.dropout)(p4)

        a5 = self.conv_block(p4, num_filter=self.num_filters * 16, kernel_size=3, batch_normalization=True)

        # Expansive Path
        b6 = Conv2DTranspose(self.num_filters * 8, (3, 3), strides=(2, 2), padding='same')(a5)
        b6 = concatenate([b6, a4])
        b6 = Dropout(self.dropout)(b6)
        a6 = self.conv_block(b6, self.num_filters * 8, kernel_size=3, batch_normalization=True)

        b7 = Conv2DTranspose(self.num_filters * 4, (3, 3), strides=(2, 2), padding='same')(a6)
        b7 = concatenate([b7, a3])
        b7 = Dropout(self.dropout)(b7)
        a7 = self.conv_block(b7, self.num_filters * 4, kernel_size=3, batch_normalization=True)

        b8 = Conv2DTranspose(self.num_filters * 2, (3, 3), strides=(2, 2), padding='same')(a7)
        b8 = concatenate([b8, a2])
        b8 = Dropout(self.dropout)(b8)
        a8 = self.conv_block(b8, self.num_filters * 2, kernel_size=3, batch_normalization=True)

        b9 = Conv2DTranspose(self.num_filters * 1, (3, 3), strides=(2, 2), padding='same')(a8)
        b9 = concatenate([b9, a1])
        b9 = Dropout(self.dropout)(b9)
        a9 = self.conv_block(b9, self.num_filters * 1, kernel_size=3, batch_normalization=True)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(a9)
        # outputs_r=np.round(outputs)
        self.model_unet = Model(inputs=self.input_image, outputs=[outputs])
        #self.model_unet.compile(optimizer='adam', loss=self.custom_loss, metrics=[self.dice_loss])
        #self.model_unet.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.BinaryCrossentropy())
        #alpha
        return self

    def dice_coeff(self,y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    def dice_loss(self,y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, np.round(y_pred))
        return loss

    def bce_dice_loss(self,y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss

    def custom_loss(self,y_true, y_pred):
        loss = np.abs(y_true - np.round(y_pred))
        return loss

    def conv_block(self, input_tensor, num_filter, kernel_size, batch_normalization=True):
        x = Conv2D(filters=num_filter, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=num_filter, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def image_to_array(self):
        X = np.zeros((1, self.img_height, self.img_width, 3), dtype=np.float32)
        img = load_img(self.test_image)
        x_img = img_to_array(img)
        x_img = resize(x_img, (self.img_height, self.img_width, 1), mode='constant', preserve_range=True)

        X[0] = x_img / 255.0
        return X

#if __name__=="__main__":
def my_main(input_img):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    obj=landsat() # create an uinstance of class

    obj.input_image = Input((obj.img_height, obj.img_width, 3), name='img')
    obj.unet_model() # create instance of model

    model=obj.model_unet
    model.load_weights("model-lndsat2.h5")

    #obj.test_image = "image_18.jpg"
    obj.test_image="static/user_uploaded_image.jpg"
    X = obj.image_to_array()
    #X=np.resize(X,[1,128,128,3])
    print(X.shape)
    pred = model.predict(X)
    pred_reshaped=np.reshape(pred, [128, 128,1])
    plt.imshow(1-np.round(pred_reshaped),cmap='gray')
    #plt.colorbar()
    plt.axis("off")
    plt.savefig("static/result_img.png")
    #plt.show()
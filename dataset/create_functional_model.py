from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Conv2D(16, (3,3), activation='relu', name="conv2d_1")(inputs)
x = MaxPooling2D(2,2)(x)
x = Conv2D(32, (3,3), activation='relu', name="conv2d_2")(x)
x = MaxPooling2D(2,2)(x)
x = Flatten()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)
model.save("model.h5")
print("Functional model.h5 created")
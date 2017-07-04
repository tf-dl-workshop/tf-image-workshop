from tensorflow.contrib.keras.python.keras.preprocessing.image import *
from tensorflow.contrib.keras.python.keras.applications.resnet50 import *
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras import backend as K

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '_data/train',
    target_size=(250, 250),
    batch_size=128)

validation_generator = test_datagen.flow_from_directory(
    '_data/validate',
    target_size=(250, 250),
    batch_size=128)

# create the base pre-trained _model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a softmax layer
predictions = Dense(5, activation='softmax')(x)

# this is the _model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the _model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy')

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# train the _model on the new _data for a few epochs
model.fit_generator(
    train_generator,
    steps_per_epoch=int(9836 / 256),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=int(2462 / 256))

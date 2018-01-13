import keras
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# Usage guide. Change layers to unfreeze to train more layers of mobile net
num_classes = 5
layers_to_unfreeze = 3
total_layers = 81

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

model = load_model('faceRec.h5', custom_objects={
    'relu6': relu6,
    'DepthwiseConv2D': DepthwiseConv2D})

for i, layer in enumerate(model.layers):
    layer.trainable = False
    if i < total_layers - layers_to_unfreeze:
        layer.trainable = False
    else:
        print(i, layer.name, ' was unfrozen')
        print('layer input shape: ', layer.input_shape, ' output shape: ', layer.output_shape);

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

training_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=12,
    class_mode='categorical',
    save_to_dir='savedFiles/training'
)

test_generator = datagen.flow_from_directory(
    'validationDataset',
    target_size=(224, 224),
    batch_size=3,
    class_mode='categorical',
    save_to_dir='savedFiles/validation'
)

# add dir. add target size.
model.fit_generator(
    training_generator,
    steps_per_epoch=18,
    epochs=5,
    validation_data=test_generator
)

scores = model.evaluate_generator(generator=test_generator)

print(model.metrics_names)

for i, name in enumerate(model.metrics_names):
    print(name, ':', scores[i])

# save Model
model.save('faceRec.h5') 


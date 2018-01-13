import keras
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

num_classes = 5

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

base = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base.output
print(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=prediction)

for i, layer in enumerate(base.layers):
    layer.trainable = False
    if i > 50 :
        print(i, layer.name)
        print('layer input shape: ', layer.input_shape, ' output shape: ', layer.output_shape);


model.compile(
    loss='categorical_crossentropy',
    #loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

training_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=6,
    class_mode='categorical',
    save_to_dir='savedFiles/training'
)

test_generator = datagen.flow_from_directory(
    'validationDataset',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    save_to_dir='savedFiles/validation'
)

# add dir. add target size.
model.fit_generator(
    training_generator,
    steps_per_epoch=18,
    epochs=4,
    validation_data=test_generator
)

scores = model.evaluate_generator(generator=test_generator)

print(model.metrics_names)

for i, name in enumerate(model.metrics_names):
    print(name, ':', scores[i])

# save Model
model.save('faceRec.h5') 


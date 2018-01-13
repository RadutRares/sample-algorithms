import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

#model = MobileNet()
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(224,224,3), padding='same', activation='relu' ))
model.add(Conv2D(16, (3, 3), input_shape=(224,224,3), padding='same', activation='relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), input_shape=(224,224,3), padding='same', activation='relu' ))
model.add(Conv2D(32, (3, 3), input_shape=(224,224,3), padding='same', activation='relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

for i, layer in enumerate(model.layers):
   print(i, layer.name)
   print('layer input shape: ', layer.input_shape, ' output shape: ', layer.output_shape);

print(model.output)

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
#	save_to_dir='savedFiles/training'
)

test_generator = datagen.flow_from_directory(
	'validationDataset',
	target_size=(224, 224),
	batch_size=1,
	class_mode='categorical',
#	save_to_dir='savedFiles/validation'
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




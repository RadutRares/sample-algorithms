import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
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
model.add(Dense(128,
	input_shape=(224,224,3),
	activation='relu'
))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='sigmoid'))

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

# See images from generator
for i, item in enumerate(test_generator):
	print(i)
	print('image', item[0].shape)
	if (i > 9):
		break

# add dir. add target size.
model.fit_generator(
	training_generator,
	steps_per_epoch=3,
	epochs=2,
	#validation_data=test_generator
)

score = model.evaluate_generator(generator=test_generator)
print('Score was ', score)
'''
'''

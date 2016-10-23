import sys

import keras
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, K
from keras.layers import Dense

import dataset

data_dir, model_file_prefix = sys.argv[1:]

n, nb_chan = 299, 3
batch_size = 128

X, y, labels = dataset.dataset(data_dir, n)
print "before preprocess", X[0].shape
#import matplotlib.pyplot as plt
#plt.imshow(X[0])
#plt.show()
X = keras.applications.inception_v3.preprocess_input(X)
print "after preprocess", X[0].shape
nb_classes = len(labels)

sample_count = len(y)
train_size = sample_count * 4 // 5
X_train = X[:train_size]
y_train = y[:train_size]
X_test  = X[train_size:]
y_test  = y[train_size:]

print ('train shape:', X_train.shape)
print ('test shape:', X_test.shape)

#if K.image_dim_ordering() == 'th':
#    X_train = X_train.reshape(X_train.shape[0], nb_chan, n, n)
#    X_test = X_test.reshape(X_test.shape[0], nb_chan, n, n)
#else:
#    X_train = X_train.reshape(X_train.shape[0], n, n, nb_chan)
#    X_test = X_test.reshape(X_test.shape[0], n, n, nb_chan)


nb_train_samples = 2000
nb_validation_samples = 400
nb_epoch = 20
#input_shape = (img_width, img_height, 3)

base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# for train, use basic aug; note rescaling is done earlier
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# for test, only rescale
test_datagen = ImageDataGenerator()#rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train,
        batch_size=batch_size, shuffle=True)

validation_generator = test_datagen.flow(X_test, y_test,
        batch_size=batch_size)

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
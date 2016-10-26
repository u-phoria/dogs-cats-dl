import sys, os, numpy as np
import h5py
from keras import optimizers
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout, K
from keras.layers import Dense

train_dir, validation_dir = sys.argv[1:]

nb_epoch = 50

# has to match
pred_batch_size = 16

# path to the model weights file.
vgg16_weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'

# dimensions of our images.
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)


def build_vgg16_conv_only_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    return model


def load_and_apply_vgg_weights(weights_path, model):
    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    with h5py.File(weights_path) as f:
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            layer = model.layers[k]
            if K.image_dim_ordering() == 'tf' and layer.__class__.__name__ in ['Convolution1D', 'Convolution2D',
                                                                               'Convolution3D', 'AtrousConvolution2D']:
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))  # convert_kernel(weights[0])
            layer.set_weights(weights)  # converted_w)


def save_bottlebeck_features(train_dir, validation_dir):
    datagen = ImageDataGenerator(rescale=1./255)

    model = build_vgg16_conv_only_model()
    load_and_apply_vgg_weights(vgg16_weights_path, model)

    print('Model loaded and weights applied.')
    model.summary()

    generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=pred_batch_size,
            class_mode=None,
            shuffle=False)
    print 'starting pred train'
    bottleneck_features_train = model.predict_generator(generator, generator.nb_sample)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_dir,
            target_size=(img_width, img_height),
            batch_size=pred_batch_size,
            class_mode=None,
            shuffle=False)
    print 'starting pred test'
    bottleneck_features_validation = model.predict_generator(generator, generator.nb_sample)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def build_classifier_top_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = build_classifier_top_model(train_data.shape[1:])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)


def finetune_top_model(train_data_dir, validation_data_dir):
    model = build_vgg16_conv_only_model()

    top_model = build_classifier_top_model(model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

save_bottlebeck_features(train_dir, validation_dir)
train_top_model()
finetune_top_model(train_dir, validation_dir)
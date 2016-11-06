import cv2
import numpy as np
import sys

from keras import optimizers
from keras.applications import VGG16
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, K
from keras.layers import Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 150, 150    #224, 224
top_model_weights_path = 'bottleneck_fc_model.h5'
finetuned_model_weights_path = 'finetuned_model.h5'


def save_bottleneck_features(train_dir, validation_dir, pred_batch_size=32):
    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
    datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    # print 'starting pred train'
    # bottleneck_features_train = model.predict(X_train, batch_size=32, verbose=1)
    # np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    #
    # print 'starting pred val'
    # bottleneck_features_validation = model.predict(X_val, batch_size=32, verbose=1)
    # np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

    model = VGG16(weights='imagenet', include_top=False)

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


def create_top_model(input_shape):
    model = Sequential()
    model.add(Flatten(name="tm_flatten", input_shape=input_shape))
    model.add(Dense(256, activation='relu', name="tm_fc0"))
    model.add(Dropout(0.5, name="tm_dropout"))
    model.add(Dense(1, activation='sigmoid', name="tm_fc1"))
    return model


def create_top_model_layers(inp, layers):
    layer_dict = dict([(layer.name, layer) for layer in layers])

    x = Flatten(name="tm_flatten")(inp)
    x.weights = layer_dict['tm_flatten'].get_weights()

    x = Dense(256, activation='relu', name="tm_fc0")(x)
    x.weights = layer_dict['tm_fc0'].get_weights()

    x = Dropout(0.5, name="tm_dropout")(x)
    x.weights = layer_dict['tm_dropout'].get_weights()

    out = Dense(1, activation='sigmoid', name="tm_fc1")(x)
    x.weights = layer_dict['tm_fc1'].get_weights()

    return out


def train_top_model(nb_epoch=50):
    # X_train_bn = np.load(open('bottleneck_features_train.npy'))
    #
    # X_val_bn = np.load(open('bottleneck_features_validation.npy'))
    #
    # model = build_classifier_top_model(X_train_bn.shape[1:])
    #
    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #
    # model.fit(X_train_bn, y_train,
    #           nb_epoch=nb_epoch, batch_size=32,
    #           validation_data=(X_val_bn, y_val))
    #
    # model.save_weights(top_model_weights_path)

    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (train_data.shape[0] / 2) + [1] * (train_data.shape[0]/ 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (validation_data.shape[0] / 2) + [1] * (validation_data.shape[0] / 2))

    print 'input shape', train_data.shape[1:]
    model = create_top_model(train_data.shape[1:])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)


def finetune_top_model(train_dir, validation_dir, nb_epoch=50, batch_size=32):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_width,img_height,3)))

    # set layers up to the last conv block to non-trainable (weights will not be updated)
    nb_non_trainable_layers = len(vgg_model.layers) - 4
    for layer in vgg_model.layers[:nb_non_trainable_layers]:
        layer.trainable = False

    #top_model = build_classifier_top_model(vgg_model.output_shape[1:])
    top_model = create_top_model(vgg_model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    out = create_top_model_layers(vgg_model.output, top_model.layers)

    model = Model(vgg_model.input, out)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # X_train, y_train, _ = dataset.dataset(train_dir, img_width, img_height,
    #                                       preprocess_fn=preprocess_input)
    # X_val, y_val, _ = dataset.dataset(validation_dir, img_width, img_height,
    #                                   preprocess_fn=preprocess_input)

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1.,
        featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    test_datagen = ImageDataGenerator(
        rescale=1.,
        featurewise_center=True)
    test_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.nb_sample,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.nb_sample)

    model.save_weights(finetuned_model_weights_path)


def predict(image_path):
    # Test pretrained model
    vgg_model = VGG16(weights=None, include_top=False, input_tensor=Input(shape=(img_width,img_height,3)))
    top_model = create_top_model(vgg_model.output_shape[1:])
    out = create_top_model_layers(vgg_model.output, top_model.layers)
    # print out.output
    model = Model(vgg_model.input, out)
    model.load_weights(finetuned_model_weights_path)

    im = cv2.resize(cv2.imread(image_path), (img_width, img_height)).astype(np.float32)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # import matplotlib.pyplot as plt
    # plt.imshow(im*1/255.0)
    # plt.show()

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    #im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    out = model.predict(im)[0]
    print out

if __name__ == "__main__":
    # print 'load train'
    # X_train, y_train, _ = dataset.dataset(train_dir, img_width, img_height,
    #                                       preprocess_fn=preprocess_input)
    #
    # print 'load val'
    # X_val, y_val, _ = dataset.dataset(validation_dir, img_width, img_height,
    #                                   preprocess_fn=preprocess_input)

    cmd = sys.argv[1]
    if cmd == 'init':
        train_dir, validation_dir = sys.argv[2:]
        save_bottleneck_features(train_dir, validation_dir)
        train_top_model()
    elif cmd == 'finetune':
        train_dir, validation_dir = sys.argv[2:]
        finetune_top_model(train_dir, validation_dir)
    elif cmd == 'pred':
        image_path = sys.argv[2]
        predict(image_path)
    else:
        raise Exception('unknown command ' + cmd)
import sys, numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

(pred_filename,) = sys.argv[1:]

model = VGG16(weights='imagenet', include_top=True)

img = image.load_img(pred_filename, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

for _, class_name, prob in decode_predictions(preds, top=5)[0]:
    print "{:.2f}: {}".format(prob, class_name)
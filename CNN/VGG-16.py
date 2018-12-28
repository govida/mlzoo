from keras.applications.vgg16 import VGG16
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

base_model = VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)

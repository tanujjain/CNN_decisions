import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import preprocessing

from utils import load_image
from utils import map_plotter


def validate_index(cnn, output_class_index):
    assert (output_class_index < cnn.output.shape[1]) and (cnn.output.shape[1] > -1), 'Provide a valid index ..'


def mobilenet_preprocess(inp_image):
    from tensorflow.keras.applications.mobilenet import preprocess_input
    image_res = inp_image.resize((224, 224))  # Input size for Mobilenet
    img_tensor = preprocessing.image.img_to_array(image_res)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    inp_to_cnn = preprocess_input(img_tensor)
    return inp_to_cnn


def _get_grads_wrt_input(cnn_model, inp_tensor, output_class_index):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inp_tensor)
        if output_class_index:
            validate_index(cnn_model, output_class_index)
            f = cnn_model(inp_tensor)[0][output_class_index]
        else:
            f = cnn_model(inp_tensor)
    grad = tape.gradient(f, inp_tensor)
    return grad


def plot_saliency_map(image_path, model_path, output_class_index, custom_preprocessing=None, sr=True):
    loaded_image = load_image(image_path)
    orig_image_size = loaded_image.size

    if not model_path:
        print('Loading Mobilenet pretrained on imagenet dataset ....')
        from tensorflow.keras.applications import MobileNet

        cnn = MobileNet()
        inp_to_cnn = mobilenet_preprocess(inp_image=loaded_image)
    else:
        from tensorflow.keras.models import load_model  # Assumption: model in h5
        cnn = load_model(model_path)
        inp_to_cnn = custom_preprocessing(inp_image=loaded_image)  #ToDo: Error handling for absence of custom_preprocessing function

    inp_tensor = tf.constant(inp_to_cnn)
    cnn_model = models.Model([cnn.inputs], [cnn.output])
    grad = _get_grads_wrt_input(cnn_model=cnn_model, inp_tensor=inp_tensor, output_class_index=output_class_index)

    if sr:
        inp_size = tuple(inp_tensor.shape[1:])  # 0-th index is for batch

        width_upsample_factor = int(np.floor(orig_image_size[0]/inp_size[0]))
        height_upsample_factor = int(np.floor(orig_image_size[1] / inp_size[1]))

        upsampling_factor = min(width_upsample_factor, height_upsample_factor)
        if upsampling_factor != 0:
            from ISR.models import RDN
            rdn = RDN(arch_params={'C': 6,
                               'D': 20,
                               'G': 64,
                               'G0': 64,
                               'x': upsampling_factor},
                      weights='psnr-small')  # 'x' is the upsampling factor. Other values are required to load
            # pre-trained model

            img_grad = preprocessing.image.array_to_img(grad[0])
            lr_img = np.array(img_grad)
            sr_img = rdn.predict(lr_img)
            full_size_salmap = tf.image.resize(sr_img, orig_image_size)  # Since upsampling factor may not fit to exact
            # original image dimensions
            sal_image = preprocessing.image.array_to_img(full_size_salmap)
    else:
        full_size_grad = tf.image.resize(grad, orig_image_size)[0]
        sal_image = preprocessing.image.array_to_img(full_size_grad)

    map_plotter(grad_image=sal_image, original_image=loaded_image)



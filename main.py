from saliency_map import plot_saliency_map

image_path = '/home/tanuj/hobby/cnn_Experiment/Images/australian-terrier-2.jpg'
model_path = None
out_ind = None


def custom_preprocessing_func():
    """Returns a function that executes preprocessing logic for images before feeding to the custom network.
    The implemented function must take exactly one argument: the input image (inp_image should be the argument name)"""
    return None


if __name__ == '__main__':
    plot_saliency_map(image_path=image_path,
                      model_path=model_path,
                      output_class_index=out_ind,
                      custom_preprocessing=custom_preprocessing_func(),
                      sr=True)

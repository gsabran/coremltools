# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys as _sys

def convert(model, image_input_names=[], is_bgr=False,
            red_bias=0.0, blue_bias=0.0, green_bias=0.0, gray_bias=0.0,
            image_scale=1.0, class_labels=None, predicted_feature_name=None):
    """
    Convert a Caffe model to Core ML format.

    Parameters
    ----------
    model: str | (str, str) | (str, str, str)

        A trained Caffe neural network model which can be represented as:

        - Path on disk to a trained Caffe model (.caffemodel)
        - A tuple of two paths, where the first path is the path to the .caffemodel
          file while the second is the path to the deploy.prototxt.
        - A tuple of three paths, where the first path is the path to the
          trained .caffemodel file, the second is the path to the
          deploy.prototxt while the third is a path to the mean image binary.

    image_input_names: [str] | str
        The name(s) of the input blob(s) in the Caffe model that can be treated
        as images by Core ML. All other inputs are treated as MultiArrays (N-D
        Arrays) by Core ML.

    is_bgr: bool
        Flag to determine if the input images are in pixel order (RGB or BGR)

    red_bias: float
        Bias value to be added to the red channel of the input image.
        Defaults to 0.0

    blue_bias: float
        Bias value to be added to the the blue channel of the input image.
        Defaults to 0.0    

    green_bias: float
        Bias value to be added to the green channel of the input image.
        Defaults to 0.0    

    gray_bias: float
		Bias value to be added to the input image (in grayscale). Defaults to 0.0    

    image_scale: float
        Value by which the input images will be scaled before bias is added and 
        Core ML model makes a prediction. Defaults to 1.0.

    class_labels: str
        Filepath where classes are parsed as a list of newline separated
        strings. Class labels map the index of the output of a neural network to labels in a classifier.
        Provide this argument to get a model of type classifier.

    predicted_feature_name: str
        Name of the output feature for the class labels exposed in the Core ML
        model (applies to classifiers only). Defaults to 'classLabel'

    Returns
    -------
    model: MLModel
        Model in Core ML format.

    Examples
    --------
    .. sourcecode:: python

		# Convert it with default input and output names
   		>>> import coremltools
		>>> coreml_model = coremltools.converters.caffe.convert('my_caffe_model.caffemodel')

		# Saving the Core ML model to a file.
		>>> coreml_model.save('my_model.mlmodel')

    Sometimes, critical information in the Caffe converter is missing from the
    .caffemodel file. This information is present in the deploy.prototxt file.
    You can provide us with both files in the conversion process.

    .. sourcecode:: python

		>>> coreml_model = coremltools.converters.caffe.convert(('my_caffe_model.caffemodel', 'my_deploy.prototxt'))

    Some models (like Resnet-50) also require a mean image file which is
    subtracted from the input image before passing through the network. This
    file can also be provided during covnersion:

    .. sourcecode:: python

        >>> coreml_model = coremltools.converters.caffe.convert(('my_caffe_model.caffemodel',
        ...                 'my_deploy.prototxt', 'mean_image.binaryproto'))
            
            
            
    Input and output names used in the interface of the converted Core ML model are inferred from the .prototxt file,
    which contains a description of the network architecture.
    Input names are read from the input layer definition in the .prototxt. By default, they are of type MultiArray.
    Argument "image_input_names" can be used to assign image type to specific inputs.
    All the blobs that are "dangling", i.e.
    which do not feed as input to any other layer are taken as outputs. The .prototxt file can be modified to specify
    custom input and output names.
    
    The converted Core ML model is of type classifier when the argument "class_labels" is specified.

    Advanced usage with custom classifiers, and images:
            
    .. sourcecode:: python

		# Mark some inputs as Images
		>>> coreml_model = coremltools.converters.caffe.convert('my_caffe_model.caffemodel',
		...                   image_input_names = 'my_image_input')

		# Export as a classifier with classes from a file
		>>> coreml_model = coremltools.converters.caffe.convert('my_caffe_model.caffemodel',
		...         image_input_names = 'my_image_input', class_labels = 'labels.txt')
            
                 
    Sometimes the converter might return a message about not able to infer input data dimensions. 
    This happens when the input size information is absent from the deploy.prototxt file. This can be easily provided by editing 
    the .prototxt in a text editor. Simply add a snippet in the beginning, similar to the following, for each of the inputs to the model:
            
    .. code-block:: bash
            
        input: "my_image_input"
        input_dim: 1
        input_dim: 3
        input_dim: 227
        input_dim: 227
                                    
    Here we have specified an input with dimensions (1,3,227,227), using Caffe's convention, in the order (batch, channel, height, width). 
    Input name string ("my_image_input") must also match the name of the input (or "bottom", as inputs are known in Caffe) of the first layer in the .prototxt.                                         

    """
    from ...models import MLModel
    import tempfile
    model_path = tempfile.mktemp()
    spec = _export(model_path, model, image_input_names, is_bgr, red_bias, blue_bias,
            green_bias, gray_bias, image_scale, class_labels,
            predicted_feature_name)
    return MLModel(model_path)


def _export(filename, model, image_input_names=[], is_bgr=False,
           red_bias=0.0, blue_bias=0.0, green_bias=0.0, gray_bias=0.0,
           image_scale=1.0,
           class_labels=None, predicted_feature_name=None):
    try:
        from ... import libcaffeconverter
    except:
        if _sys.platform != 'darwin':
            raise RuntimeError('Caffe conversion is only supported on macOS.')
        else:
            raise RuntimeError('Unable to load Caffe converter library.')
    if isinstance(model, basestring):
        src_model_path = model
        prototxt_path = u''
        binaryproto_path = u''
    elif isinstance(model, tuple):
        if len(model) == 3:
            src_model_path, prototxt_path, binaryproto_path = model
        else:
            src_model_path, prototxt_path = model
            binaryproto_path = u''


    if isinstance(image_input_names, basestring):
        image_input_names = [image_input_names]
    if predicted_feature_name is None:
        predicted_feature_name = u'classLabel'
    if class_labels is None:
        class_labels = u''
    libcaffeconverter._convert_to_file(src_model_path,
                                       filename,
                                       set(image_input_names),
                                       is_bgr,
                                       red_bias,
                                       blue_bias,
                                       green_bias,
                                       gray_bias,
                                       image_scale,
                                       prototxt_path,
                                       binaryproto_path,
                                       class_labels,
                                       predicted_feature_name
                                       )

# Zero-Shot Ultrasound Nondestructive Testing Image Super-Resolution Based on Reflection Projection

----------------------------------------------------------------------------------------------------
In ultrasonic nondestructive testing, the low resolution of ultrasound images possibly lead 
to misinterpretation of defects in the image. At present, there is no spe-cial data set for
ultrasonic nondestructive testing images in super-resolution, and the performance of numerous
existing models depends on the learning of general data sets. In this paper, a zero-shot 
super-resolution network based on reflection projection units is proposed. Ultrasound images
contain numerous image blocks with similar content, which are randomly extracted and down-sampled 
to form training samples. Then the image features are extracted through the reflection projection
units in the network, and the information between the high and low-resolution image pairs is
fully excavated. Finally, the feature channel is reduced by the attention mechanism, and the
reconstructed image is output. Moreover, a combined loss function is used to optimize the 
network parameters. The com-pared experiments show that the proposed method performs better 
than the state of the art.
-----------------------------------------------------------------------------------------------------

# If you need to use this code, just change the relevant path. Fine tuning parameters may make your experiment better.

------------------------------------------------------------------------------------------------------
The contribution of this paper benefits from “Zero-Shot” Super-Resolution using Deep Internal Learning"
You can modify relevant parameters in "configs.py" to meet your needs.
such as 
    result_path = os.path.dirname(__file__) + '/results'
    input_path = local_dir = os.path.dirname(__file__) + '/input'
-------------------------------------------------------------------------------------------------------
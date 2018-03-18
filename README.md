# UniversalStyleTransfer
Universal Style Transfer via Feature Transforms using TensorFlow

The paper published by Li et al named “Universal Style Transfer via Feature Transforms” (https://arxiv.org/pdf/1705.08086.pdf) is implemented using TensorFlow. 

The entire project is based on an auto-encoder trained model that allows rebuilding the output image from intermediate layers of already trained VGG19 image classification net. 

In order to obtain stylization, it is necessary to map the statistical features of content image with that of the desired style image. This is done through the Whitening-Coloring transform model.

There is no requirement of training style images thus permitting the WCT to perform 'universal' style transfer for any random content-style image pairs.

This paper involves reconstructing the decoders for all the relu_layers(X=1,2,3,4,5). Initially, training is performed individually on each of these layers. Later, they are held together in a multi-level stylization pipeline.

A single VGG encoder is set up to the deepest relu_layer which is been shared by all the decoders, promoting reduced memory usage.


# Samples:

Content Image
Style Image
Result

![imageres](https://clemson.box.com/shared/static/spxeli5sw26z8r8t34ndsi9ccr1twc9n.jpg?raw=true "Title")



# Requirements:

* Python 3.5
* TensorFlow 1.5.0
* Palmetto Cluster 
* Anaconda3 5.0.1
* Keras


# Libraries Used:

* scikit-image
* ffmpeg
* numpy
* librosa


# Running the model:

At first, we need to download the pre-trained models, that is VGG19 model given by download_vgg.sh from https://www.dropbox.com/s/ssg39coiih5hjzz/models.zip?dl=1 and the checkpoints for the decoders given by download_models.sh from https://www.dropbox.com/s/kh8izr3fkvhitfn/vgg_normalised.t7?dl=1


3. Input the desired content images and the style images on which you wish to run the model.

4.Run the stylize.py using the following command:
 
python stylize.py --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 --content-path <CONTENT IMAGE PATH>
--style-size 512 --alpha 0.8 --style-path <STYLE IMAGE PATH> --out-path <OUTPUT IMAGE PATH> 

# Arguments used are:

1. checkpoints:  variable to restore to specific checkpoint.
2. relu-targets: mapping the checkpoints to corresponding relu_layer 
                 targets.
3. content-path: path of the content image or folder containing the    
                 content image.
4. alpha:        weighing factor that is the WCT feature to control 
                 degree of stylization.
5. style-size:   resizing the original image to the assigned value
6. style-path:   path of the style image or folder containing the    
                 style image.
7. out-path:     path of the folder where you want to save the    
                 output image.

# Addition to existing model:

Here we are implementing the Style Transfer model not only to Images but also to Audio Files. So, we input a content ‘mp3’ file and a style ‘mp3’ file. 

Modifying the static methods, preprocess(object) and postprocess(object) in WCT.py file to implement the above Universal Style Transfer model on the audio files.


In case you want to use your own audio files as inputs, 
We need to first cut them to 10s length using the code: 

ffmpeg -i <FILENAME.mp3> -ss 00:00:00 -t 10 <FILENAME.mp3>

Using Fast Fourier Transform, we preprocess the content and style ‘mp3’ files and then we apply the Universal Style Transform model that is mentioned above to transfer style of the style ‘mp3’ file to the content ‘mp3’ file and then carry out reconstruction to obtain the output .wav file.



# References:

 1. Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, Ming-Hsuan Yang. Universal   Style Transfer via Feature Transforms. In 
2. L. A. Gatys, A. S. Ecker, and M. Bethge. Texture synthesis using convolutional neural networks. In NIPS, 2015.
3. L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. In CVPR, 2016.
4. X. Huang and S. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, 2017.
5. J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016.
6. C. Li and M. Wand. Precomputed real-time texture synthesis with markovian generative adversarial networks. In ECCV, 2016.
 7. D. Ulyanov, V. Lebedev, A. Vedaldi, and V. Lempitsky. Texture networks: Feed-forward synthesis of textures and stylized images. In ICML, 2016.
8. Eric Grinstein, Ngoc Duong, Alexey Ozerov, Patrick Perez, Audio Style Transfer.In 2017.



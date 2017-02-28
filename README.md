# Colorize Your World with Deep Neural Network
God said, "let there be color"; he willed it, and at once the *Deep Neural Network* brings the color!

Here is [an intuitive demo](https://github.com/Lyken17/Colorize-Your-World/blob/master/Demo.ipynb).

This project is based on the [siggraph16](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf) paper. Please refer to this paper for the details of the model.

Our implementation is slightly different from the one proposed in the aforementioned paper. The experiments show that our implementation is much more efficient and requires significantly less graphic memory in the training phase compared to the implementation proposed by the paper.

# Colorize your favorite grayscale images
* Hardware requirements

  If you want to train a model on your own dataset, a powerful GPU is required (at least 6GB graphic memory). If you just want to play with it, a CPU would be enough and you should download our pre-trained models.

* Install the framework

  The implementation of this project is on the [Torch7](http://torch.ch), a popular deep learning framework. Follow the [install guide](http://torch.ch/docs/getting-started.html#) to install this framework on your computer.

* Setup dependencies
  ```
  luarocks install torch
  luarocks install nn
  luarocks install image
  luarocks install lua-cjson
  luarocks install hdf5

  #GPU acceleration
  luarocks install cutorch
  luarocks install cunn
  luarocks install cudnn
  ```
  
*  Download pre-trained model (if you don't want to train your own model)

    `wget https://github.com/Lyken17/Colorize-Your-World/releases/download/1.0/pre_trained.t7`

* Colorize your favorite images

    `th colorzie.lua -input_image 'your image' `

# Train your own model

The pre-trained model we provide is carefully finetuned and should work well for general purposes. But if you want to train your own model anyway, you can use the following command (with the default training parameters):
  
  `th train.lua -h5_file 'your training database here'`
  
 You can specify many training parameters. Please refer to `train.lua` for a full list of parameters.

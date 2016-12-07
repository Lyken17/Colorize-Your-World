# Colorize Your World with Deep Neural Network
God said, "let there be color"; he willed it, and at once *Deep Neural Network* brings the  color!

[An intuitive demo](https://github.com/Lyken17/Colorize-Your-World/blob/master/Demo.ipynb)

# Notice
This work is based on  [siggraph16](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf) with several modifications. If you wonder details about the model, peruse the original paper would be a good choice.

# How to run this code
* Hardwares

  If you are going to train on your own dataset, a fancy GPU is necessary (at least 6GB graphic memory). If you are going to just play with it, then CPU would be enough to handle.

* Framework

  Code is based on [Torch7](http://torch.ch). Follow [install guide](http://torch.ch/docs/getting-started.html#) first.

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

# Colorize  images
*  Download pre-trained model (if you are not going to train your own)

    `wget https://github.com/Lyken17/Colorize-Your-World/releases/download/1.0/pre_trained.t7`

* Magic Time

    `th colorzie.lua -input_image 'your image' `

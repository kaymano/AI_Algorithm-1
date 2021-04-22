# AI_Algorithm-1
## Installation
First, run `pip install pipenv`
Then, run `pipenv --python path/to/64-bit/python.exe install` where `path/to/64-bit/python.exe` is the path to a 64 bit python installation on your machine. If the only python installation you have is 64 bit, the you can just run `pipenv install`.
If you get an error installing tensorflow, it is likely you are not using a 64 bit python.
Next, run `pipenv shell` and then `python -m pip install ./object_detection`
If this fails, go to the [TensorFlow Object Detection API Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) and [Pytorch Documentation](https://pytorch.org/get-started/locally/) and follow the installation instructions there.
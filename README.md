# AI_Algorithm-1
## Installation
First, run `pip install pipenv`
Then, run `pipenv --python path/to/64-bit/python.exe install` where `path/to/64-bit/python.exe` is the path to a 64 bit python installation on your machine. If the only python installation you have is 64 bit, the you can just run `pipenv install`.
If you get an error installing tensorflow, it is likely you are not using a 64 bit python.
Next, run `pipenv shell` and then `python -m pip install ./models/research` . Then run `pip install numpy==1.17` and ignore the error. You should now be able to run `python models\research\object_detection\model_main_tf2.py --model_dir=wally_model_v6 --pipeline_config_path=wally_model_v6\pipeline.config --checkpoint_dir=wally_model_v6` to test the model on the training data and `python models\research\object_detection\model_main_tf2.py --model_dir=wally_model_v6 --pipeline_config_path=exported_models\wally_model_v6\pipeline.config --checkpoint_dir=wally_model_v6` to test the model on the training data. 
If this fails, go to the [TensorFlow Object Detection API Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) and [Pytorch Documentation](https://pytorch.org/get-started/locally/) and follow the installation instructions there.

# Where's Waldo AI 
This program runs the Object Detection code to find Waldo from 10 images. 

## Contributors: Team 5
Adam Sickels, Kayla Manouchehri, Abha Naik, Sery Gunawardena, Chase 

## Table of Contents:
- Repository Setup
- Gem Installations
- Classes
- Program Instructions 
- Testing

## Repository Setup:

Clone repository by running 'git clone https://github.com/cse3901-2020au-1730/SS-PBJ-Project3.git' in terminal

## Installations:

1. Install pipenv
```bash
pip install pipenv
```
2. If you do not have 64-bit python, run
```bash
pipenv --python path/to/64-bit/python.exe install
```

Otherwise, run
```bash
pipenv install
```
3. 
```bash
pipenv shell
```

4. 
```bash
python -m pip install ./models/research
```

5. 
Install numpy and ignore the error. 
```bash
pip install numpy==1.17
```

## Program Instructions

Test the model 
```bash
python models\research\object_detection\model_main_tf2.py --model_dir=wally_model_v6 --pipeline_config_path=wally_model_v6\pipeline.config --checkpoint_dir=wally_model_v6
```

```bash
python models\research\object_detection\model_main_tf2.py --model_dir=wally_model_v6 --pipeline_config_path=exported_models\wally_model_v6\pipeline.config --checkpoint_dir=wally_model_v6
```
First, run `pip install pipenv`
Then, run `pipenv --python path/to/64-bit/python.exe install` where `path/to/64-bit/python.exe` is the path to a 64 bit python installation on your machine. If the only python installation you have is 64 bit, the you can just run `pipenv install`.
If you get an error installing tensorflow, it is likely you are not using a 64 bit python.
Next, run `pipenv shell` and then `python -m pip install ./models/research` . Then run `pip install numpy==1.17` and ignore the error. You should now be able to run `python models\research\object_detection\model_main_tf2.py --model_dir=wally_model_v6 --pipeline_config_path=wally_model_v6\pipeline.config --checkpoint_dir=wally_model_v6` to test the model on the training data and `python models\research\object_detection\model_main_tf2.py --model_dir=wally_model_v6 --pipeline_config_path=exported_models\wally_model_v6\pipeline.config --checkpoint_dir=wally_model_v6` to test the model on the training data. 
If this fails, go to the [TensorFlow Object Detection API Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) and [Pytorch Documentation](https://pytorch.org/get-started/locally/) and follow the installation instructions there.



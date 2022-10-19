# Comp411 HW1

This assignment is adapted from [Stanford Course CS231n](http://cs231n.stanford.edu/).

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor, the SVM/Softmax classifier and a simple Neural Network classifier. The goals of this assignment are as follows:

- understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
- understand the train/val/test splits and the use of validation data for hyperparameter tuning.
- develop proficiency in writing efficient vectorized code with numpy
- implement and apply a k-Nearest Neighbor (kNN) classifier
- implement and apply a Multiclass Support Vector Machine (SVM) classifier
- implement and apply a Softmax classifier
- implement and apply a Four layer neural network classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

## Setup Instructions


**Installing Anaconda:** If you decide to work locally, we recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.7. We are no longer supporting Python 2.

**Anaconda Virtual environment:** Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

`conda create -n comp411 python=3.7 anaconda=2021.05`

to create an environment called comp411.

Then, to activate and enter the environment, run

`conda activate comp411`

To exit, you can simply close the window, or run

`conda deactivate comp411`

Note that every time you want to work on the assignment, you should run `conda activate comp411` (change to the name of your virtual env).

You may refer to [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.



## Installing packages:

Once you’ve setup and activated your virtual environment (via conda), you should install the libraries needed to run the assignments using pip. To do so, run:

```
# again, ensure your virtual env (conda)
# has been activated before running the commands below:

cd comp411_assignment1_2020  # cd to the assignment directory

# install assignment dependencies.
# since the virtual env is activated,
# this pip is associated with the
# python binary of the environment

pip install -r requirements.txt
```


In order to use the correct version of `scipy` for this assignment,
run the following after your first activation of the environment:

`pip install scipy=1.1.0`



## Download data:

Once you have the starter code, you will need to download the CIFAR-10 dataset. Make sure `wget` is installed on your machine before running the commands below. Run the following from the assignment1 directory:

```
cd comp411/datasets
./get_datasets.sh
```

## Start IPython:

After you have the CIFAR-10 data, you should start the IPython notebook server from the assignment1 directory, with the jupyter notebook command.

If you are unfamiliar with IPython, you can also refer to [IPython tutorial](http://cs231n.github.io/ipython-tutorial/).

## Grading

Q1: k-Nearest Neighbor classifier (25 points)
    -- Please follow the Jupyter Notebook, "knn.ipynb" to complete this part of the assignment

Q2: Training a Support Vector Machine (20 points)
    -- Please follow the Jupyter Notebook, "svm.ipynb" to complete this part of the assignment

Q3: Implement a Softmax classifier (20 points)
    -- Please follow the Jupyter Notebook, "softmax.ipynb" to complete this part of the assignment

Q4: Four-Layer Neural Network (25 points)
    -- Please follow the Jupyter Notebook, "four_layer_net.ipynb" to complete this part of the assignment

Q5: Higher Level Representations: Image Features (10 points)
    -- Please follow the Jupyter Notebook, "features.ipynb" to complete this part of the assignment

## Submission

Zip (do not use RAR) the assignment folder using the format `username_studentid_assignment1.zip`.
Email the zip file to the instructor and TA. Do not include large files in the submission (for
instance data files under `./comp411/datasets/cifar-10-batches-py`).

## Notes

NOTE 1: Make sure that your homework runs successfully. Otherwise, you may get a zero grade from the assignment.

NOTE 2: There are # *****START OF YOUR CODE/# *****END OF YOUR CODE tags denoting the start and end of code sections you should fill out. Take care to not delete or modify these tags, or your assignment may not be properly graded.

NOTE 3: The assignment1 code has been tested to be compatible with python version 3.7 (it may work with other versions of 3.x, but we won’t be officially supporting them). You will need to make sure that during your virtual environment setup that the correct version of python is used. You can confirm your python version by (1) activating your virtualenv and (2) running which python.

NOTE 4: If you are working in a virtual environment on OSX, you may potentially encounter errors with matplotlib due to the [issues described here](https://matplotlib.org/faq/virtualenv_faq.html). In our testing, it seems that this issue is no longer present with the most recent version of matplotlib, but if you do end up running into this issue you may have to use the start_ipython_osx.sh script from the assignment1 directory (instead of jupyter notebook above) to launch your IPython notebook server. Note that you may have to modify some variables within the script to match your version of python/installation directory. The script assumes that your virtual environment is named .env.

## Troubleshooting

**macOS**

If you are having problems with matplotlib (e.g. imshow), try running this:

`conda install python.app`


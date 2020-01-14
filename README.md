# AMLS_assignment
Applied Machine Learning System (ELEC0134) Assignment
Student Number: 16002077

Required Python3 version 3.6.9

**Library required**
- Keras (v 2.3.1)
- Keras-Applications (v 1.0.8)
- Keras-Preprocessing (v 1.1.0)
- Matplotlib (v 3.1.2)
- Numpy (v 1.18.0)
- OpenCV-python (v 4.1.2.30)
- Pandas (v 0.25.3)
- scikit-learn (0.22.1)
- Tensorflow (v 2.0.0)
- dlib (v 19.19.0)

**Instruction for installing the packages**
Use Pip to install required packages:
  python get-pip.py
  python -m pip install --upgrade pip
  pip install <PACKAGE>

## preprocess.py
Contains preprocessing function for task A1, A2, B1 and B2

## lab2_landmarks.py
Contains facial landmarks extraction function for task A1 and A2.
Includes several parameters for customization.
| Parameter | Description  |
| --------- | -----------  |
|  basedir | AMLS_assignment directory |
| images_dir | Image folder within dataset directory |
| labels_filename | Labels atrribute CSV within dataset directory |
| CNN_model_path | Directory store CNN model |

## main.py
Main file to run.

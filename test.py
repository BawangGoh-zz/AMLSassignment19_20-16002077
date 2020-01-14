import preprocess
from lab2_landmarks import *
from A1.task_A1 import SVM_A1
# from A2.task_A2 import SVM_A2
from B1.task_B1 import CNN_B1


# # ======================================================================================================================
# Task A1
# ======================================================================================================================
# Data preprocessing (the validation set will split automatically in scikit-learn cross-validation function)
train_X, test_X, train_Y, test_Y = preprocess.data_preprocessing_A1(images_dir, celeba_dir, labels_filename)

# Preprocessing extra test dataset 
extra_test_X, extra_test_Y = preprocess.extra_preprocessing_A1(images_test_dir, celeba_test_dir, labels_test_filename)

# Build SVM model
model_A1 = SVM_A1()                 # Build model object.
acc_A1_train, SVM_A1_clf = model_A1.train(train_X, train_Y, test_X, test_Y) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test(SVM_A1_clf, extra_test_X, extra_test_Y)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================
# Task A2
# ======================================================================================================================
# Data preprocessing (the validation set will split automatically in scikit-learn cross-validation function)
train_X2, test_X2, train_Y2, test_Y2 = preprocess.data_preprocessing_A2(images_dir, celeba_dir, labels_filename)

# # Preprocessing extra test dataset 
extra_test_X2, extra_test_Y2 = preprocess.extra_preprocessing_A2(images_test_dir, celeba_test_dir, labels_test_filename)

# Build SVM model
model_A2 = SVM_A2()                 # Build model object.
acc_A2_train, SVM_A2_clf = model_A2.train(train_X2, train_Y2, test_X2, test_Y2) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A2_test = model_A2.test(SVM_A2_clf, extra_test_X2, extra_test_Y2)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================
# Task B1
# ======================================================================================================================
# Preprocessing training dataset 
train_gen, valid_gen, eval_gen, test_gen = preprocess.data_preprocessing_B1(cartoon_images_dir, labels_path)
model_B1 = CNN_B1()
acc_B1_train, model_path1 = model_B1.train(B1_dir, 5, train_gen, valid_gen, eval_gen)
acc_B1_test = model_B1.test(model_B1_path, test_gen)
# Clean up memory/GPU etc...


# # ======================================================================================================================
# # Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# # Clean up memory/GPU etc...


# # ======================================================================================================================
# ## Print out your results with following format:
def print_train_test_acc(task, dct1, dct2):
	print(task + 'train accuracy: ')
	for item, value in dct1.items():
		print('{}: ({})'.format(item, value))

	print(task + 'test accuracy: ')
	for item, value in dct2.items():
		print('{} ({})'.format(item, value))

print_train_test_acc('Task A1', acc_A1_train, acc_A1_test)
print_train_test_acc('Task A1', acc_A2_train, acc_A2_test)
print_train_test_acc('Task B1', acc_B1_train, acc_B1_test)


# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A1_train = 'TBD'
# # acc_A1_test = 'TBD'
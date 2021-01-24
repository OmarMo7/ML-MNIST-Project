from mlxtend.data import loadlocal_mnist
from skimage.feature import hog
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, precision_score, recall_score
################### Loading Data #################
(train_x, train_y) = loadlocal_mnist(
    images_path='E:/Computer Scihhh/Year 3 - Term 1/Ml/working_project/dataset/MNIST/train-images.idx3-ubyte', labels_path='E:/Computer Scihhh/Year 3 - Term 1/Ml/working_project/dataset/MNIST/train-labels.idx1-ubyte')

(test_x, test_y) = loadlocal_mnist(
    images_path='E:/Computer Scihhh/Year 3 - Term 1/Ml/working_project/dataset/MNIST/t10k-images.idx3-ubyte', labels_path='E:/Computer Scihhh/Year 3 - Term 1/Ml/working_project/dataset/MNIST/t10k-labels.idx1-ubyte')


#####################  HOG  ####################
list_hog_fd = []
list_hog_image = []
for feature in train_x:
    fd, hog_image = hog(feature.reshape(28, 28), orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
    list_hog_fd.append(fd)
    list_hog_image.append(hog_image)
hog_features = np.array(list_hog_fd)


t_list_hog_fd = []
t_list_hog_image = []
for test_feature in test_x:
    t_fd, t_hog_image = hog(test_feature.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
    t_list_hog_fd.append(t_fd)
    t_list_hog_image.append(t_hog_image)
t_hog_features = np.array(t_list_hog_fd)


##################### KNN ##################
classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
classifier.fit(hog_features, train_y)
knn_prediction = classifier.predict(t_hog_features)


##################### SVM ##################
SVM = svm.SVC()
SVM.fit(hog_features, train_y)
svm_prediction = SVM.predict(t_hog_features)


##################### ANN ##################
model = Sequential()
model.add(Dense(392, input_dim=392, activation='relu'))
model.add(Dense(56, activation='relu'))  # 56
model.add(Dense(28, activation='relu'))  # 28
model.add(Dense(14, activation='relu'))  # 14
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(hog_features, train_y, epochs=10)
ann_prediction = model.predict(t_hog_features)
ann_prediction = np.argmax(ann_prediction, axis=1)

############## Evaluation & Comparisons #####################


def evaluation_accuracies(labels_list, prediction_list):
    print("Accuracy_Score =", accuracy_score(
        labels_list, prediction_list) * 100, "%")
    print("R2_score =", r2_score(labels_list, prediction_list) * 100, "%")
    print("Confusion matrix =", confusion_matrix(labels_list, prediction_list))
    print("precision =", precision_score(
        labels_list, prediction_list, average=None) * 100)
    print("Recall =", recall_score(labels_list,
                                   prediction_list, average=None) * 100)
    print(classification_report(labels_list, prediction_list))


def comparisons(labels_list, knn_prediction, svm_prediction, ann_prediction):
    print("KNN accuracies: \n")
    evaluation_accuracies(labels_list, knn_prediction)
    print("--------------------------------------------")
    print("SVM accuracies: \n")
    evaluation_accuracies(labels_list, svm_prediction)
    print("--------------------------------------------")
    print("ANN accuracies: \n")
    evaluation_accuracies(labels_list, ann_prediction)


comparisons(test_y, knn_prediction, svm_prediction, ann_prediction)

# three reasons to separate
# one: best practise
# two: ease to browse the code
# three: to perform unit test

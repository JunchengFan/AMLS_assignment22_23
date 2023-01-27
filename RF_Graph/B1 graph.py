import PIL.Image
import numpy as np
import sklearn
import cv2
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt



# Load the dataset

images = []
labels = []
with open ('/Users/fanjuncheng/Desktop/dataset_AMLS_22-23/cartoon_set/labels.csv') as f:
    f.readline() #to skip the header
    for line in f:
        idx, label1, label2, name = line.strip().split('\t')
        image = cv2.imread('/Users/fanjuncheng/Desktop/dataset_AMLS_22-23/cartoon_set/img/'+name) # images are read
        image = cv2.resize(image, (150, 150))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)
        if int(label2) == 0:
            labels.append(0) #face shape 0
        elif int(label2) == 1:
            labels.append(1) # face shape 1
        elif int(label2) == 2:
            labels.append(2) #face shape 2
        elif int(label2) == 3:
            labels.append(3) #face shape 3
        elif int(label2) == 4:
            labels.append(4) #face shape 4
        #print(name, labels)

# Split the data into training and test sets
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=21)

#covert list to arrays
X_train = np.array(images_train, dtype=np.float32)
X_test = np.array(images_test, dtype=np.float32)
y_train = np.array(labels_train)
y_test = np.array(labels_test)

X_train = np.array(images_train).reshape(len(images_train), -1)
X_test = np.array(images_test).reshape(len(images_test), -1)

depths = list(range(3,51))

training_acc=[]
test_acc=[]
# Create a random forest classifier
for depth in depths:
    clf = RandomForestClassifier(n_estimators=20, random_state=21, max_depth=depth)
    clf.fit(X_train, y_train)

    training_acc.append(clf.score(X_train, y_train))
    print("training_acc=", training_acc)
    labels_pred = clf.predict(X_test)
    test_acc.append(accuracy_score(y_test, labels_pred))

    print('Accuracy: ', test_acc)


plt.plot(depths, training_acc, label = "B1 Training accuracy")
plt.plot(depths, test_acc, label = 'B1 Test accuracy')
plt.xlabel("Max tree depth")
plt.ylabel('Accuracy')
plt.legend()
plt.show()




import numpy as np
import sklearn
import cv2
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



# Load the dataset

images = []
labels = []
with open ('/Users/fanjuncheng/Desktop/dataset_AMLS_22-23/celeba/labels.csv') as f:
    f.readline() #to skip the header
    for line in f:
        idx, name, label1, label2 = line.strip().split('\t')
        image = cv2.imread('/Users/fanjuncheng/Desktop/dataset_AMLS_22-23/celeba/img/'+name) # images are read
        image = cv2.resize(image, (150, 150))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)
        if int(label2) == -1:   #-1 means not smiling
            labels.append(0)
        else:
            labels.append(1)
        #print(name, labels)

print(np.array(labels).shape)
print(np.array(image).shape)

# Split the data into training and test sets
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=21)

#covert list to arrays
X_train = np.array(images_train)
X_test = np.array(images_test)
y_train = np.array(labels_train)
y_test = np.array(labels_test)

X_train = np.array(images_train).reshape(len(images_train), -1)
X_test = np.array(images_test).reshape(len(images_test), -1)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=21, max_depth=10)

# Train the classifier on the training data
clf.fit(X_train, y_train)
print("training_acc=", clf.score(X_train, y_train))

# Make predictions on the test data
labels_pred = clf.predict(X_test)

# Calculate the accuracy of the predictions
acc = accuracy_score(y_test, labels_pred, normalize=True)
print('Accuracy: ', acc)

importances = clf.feature_importances_

# Plot the feature importances
import matplotlib.pyplot as plt

plt.bar(range(X_train.shape[1]), importances)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()


from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf.estimators_[0], out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

from PIL import Image
graph.write_png('graph.png')
plt.figure(figsize=(20, 20), dpi=300)
plt.imshow(Image.open('graph.png'))
img = Image.open('graph.png')
img.save('A2_graph_high_resol_depth=3.png', dpi=(300, 300))

import sklearn.metrics as metrics

#Get precision, recall, F1 score, and confusion matrix

precision = metrics.precision_score(y_test, labels_pred, average='micro')
recall = metrics.recall_score(y_test, labels_pred, average='micro')
f1 = metrics.f1_score(y_test, labels_pred, average='micro')
confusion_matrix = metrics.confusion_matrix(y_test, labels_pred, normalize='true')

print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
print('Confusion Matrix: ', confusion_matrix)
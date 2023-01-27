import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,precision_recall_curve
import numpy as np
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        plt.savefig('acc-loss.png')


def main():
    # Check if GPU is available
    if tf.test.is_gpu_available():
        # Use GPU
        with tf.device('/device:GPU:0'):
            # Read labels
            images = []
            labels = []
            with open('../Datasets/cartoon_set/labels.csv') as f:
                f.readline()  # Skip header
                for line in f:
                    idx, label_eye, label_face, name = line.strip().split('\t')
                    # Read image
                    image = cv2.imread('../Datasets/cartoon_set/img/'+name)
                    # Resize image
                    image = cv2.resize(image, (150, 150))
                    # Convert channels to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    images.append(image)

                    labels.append(label_eye)
                    #print(name,label_eye)

            # Read test labels
            images_test = []
            labels_test = []
            with open('../Datasets/cartoon_set_test/labels.csv') as f_test:
                f_test.readline()  # Skip header
                for line_test in f_test:
                    idx_test, label_eye_test, label_face_test, name_test = line_test.strip().split('\t')
                    # Read image
                    image_test = cv2.imread('../Datasets/cartoon_set_test/img/'+name_test)
                    # Resize image
                    image_test = cv2.resize(image_test, (150, 150))
                    # Convert channels to RGB
                    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
                    images_test.append(image_test)
                    labels_test.append(label_eye_test)
            images_test = np.array(images_test)     
            labels_test = np.array(labels_test)  

            # Split data into train, val, and test sets
            x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
            # Convert list to array
            x_train = np.array(x_train)
            x_val = np.array(x_val)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_val = np.array(y_val)
            y_test = np.array(y_test)

            # Preprocessing for training data
            train_datagen = ImageDataGenerator(rescale=1./255)

            # Preprocessing for validation and test data
            val_datagen = ImageDataGenerator(rescale=1./255)

            # Load training data
            train_generator = train_datagen.flow(x_train, y_train, batch_size=48)

            # Load validation data
            val_generator = val_datagen.flow(x_val, y_val, batch_size=48)

            # Load test data
            test_generator = val_datagen.flow(x_test, y_test, batch_size=48)

            # # Build model
            model = Sequential()
            # 添加卷积层
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))


            # 添加全连接层
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax'))

            
            model.summary()  #打印出模型概况，它实际调用的是keras.utils.print_summary

            model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07, decay=0.0),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = LossHistory()
            
            model.fit_generator(
                train_generator,
                steps_per_epoch=200,
                epochs=20,
                validation_data=val_generator,
                validation_steps=50,
                callbacks=[history])
            if os.path.exists('./save_model/model.h5'):
                pass
            else:
                os.makedirs('./save_model')
                model.save('./save_model/model.h5')
            #load model to test
            #precision recall F1
            y_pred = model.predict(images_test) # 预测结果
            y_pred = np.argmax(y_pred, axis=1)            
            y_pred = y_pred.astype(int)
            print('__y_pred:__',y_pred)
            y_true = labels_test
            y_true = y_true.astype(int)
            print('__y_true:__',y_true)
            #precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            # for i, threshold in enumerate(thresholds):
            #     # 预测类别
            #     y_pred = (y_pred > threshold).astype(int)
            #     # 输出阈值和对应的精度和召回率
            #     print("Threshold:", threshold, "Precision:", precision[i], "Recall:", recall[i])
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred,average='macro')
            recall = recall_score(y_true, y_pred,average='macro')
            f1 = f1_score(y_true, y_pred,average='macro')

            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('P-R Curve')
            plt.savefig('PR.png')
            
            print(classification_report(y_true, y_pred))
            print('Test accuracy:', accuracy)
        
            history.loss_plot('epoch')         

if __name__=='__main__':
    main()
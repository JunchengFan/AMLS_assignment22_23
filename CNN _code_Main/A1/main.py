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
            with open('../Datasets/celeba/labels.csv') as f:
                f.readline()  # Skip header
                for line in f:
                    idx, name, label1, label2 = line.strip().split('\t')
                    # Read image
                    image = cv2.imread('../Datasets/celeba/img/'+name)
                    # Resize image
                    image = cv2.resize(image, (150, 150))
                    # Convert channels to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    if int(label1) == -1:
                        labels.append(0)
                    else:
                        labels.append(1)
                    #print(name,label1)

            # Read test labels
            images_test = []
            labels_test = []
            with open('../Datasets/celeba_test/labels.csv') as f_test:
                f_test.readline()  # Skip header
                for line_test in f_test:
                    idx_test, name_test, label1_test, label2_test = line_test.strip().split('\t')
                    # Read image
                    image_test = cv2.imread('../Datasets/celeba_test/img/'+name_test)
                    # Resize image
                    image_test = cv2.resize(image_test, (150, 150))
                    # Convert channels to RGB
                    image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
                    images_test.append(image_test)
                    if int(label1_test) == -1:
                        labels_test.append(0)
                    else:
                        labels_test.append(1)
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
            train_generator = train_datagen.flow(x_train, y_train, batch_size=20)

            # Load validation data
            val_generator = val_datagen.flow(x_val, y_val, batch_size=20)

            # Load test data
            test_generator = val_datagen.flow(x_test, y_test, batch_size=20)

            # # Build model
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu',#filter（卷积核）大小3*3，卷积层输出的特征图为32个，relu是激活函数的一种。很多地方说relu函数的公式就是：f(x)=max(0,x) 
                                    input_shape=(150, 150, 3)))   #输入的图片大小是150*150，3表示图片像素用（R，G，B表示）
            model.add(MaxPooling2D((2, 2)))   #空域信号施加最大值池化，（2，2）将使图片在两个维度上均变为原长的一半
            
            #第二个卷积层，64个卷积核，每个卷积核大小3*3。
            #激活函数用relu
            #还可以在Activation('relu')后加上dropout，防止过拟合
            #采用maxpooling，poolsize为(2,2)
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            
            #第三、四个卷积层，128个卷积核，每个卷积核大小3*3。
            #激活函数用relu
            #还可以在Activation('relu')后加上dropout，防止过拟合
            #采用maxpooling，poolsize为(2,2)
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            
            #全连接层，先将前一层输出的二维特征图flatten为一维的。
            #全连接有512个神经元节点,初始化方式为normal，激活函数是relu
            model.add(Flatten())  #全连接展平,即把输出的多维向量压扁后，传到普通层。压平即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
            model.add(Dense(512, activation='relu')) #全连接层，神经元个数为512  dense是导入bp层
            model.add(Dense(1, activation='sigmoid')) #激活函数是sigmoid，输出是2分类
            
            model.summary()  #打印出模型概况，它实际调用的是keras.utils.print_summary

            model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07, decay=0.0),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = LossHistory()
            
            model.fit_generator(
                train_generator,
                steps_per_epoch=100,
                epochs=30,
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
            y_pred = np.trunc(y_pred)
            y_pred = y_pred.astype(int)
            print('__y_pred:__',y_pred)
            y_true = labels_test
            #print('__y_true:__',y_true)
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            # for i, threshold in enumerate(thresholds):
            #     # 预测类别
            #     y_pred = (y_pred > threshold).astype(int)
            #     # 输出阈值和对应的精度和召回率
            #     print("Threshold:", threshold, "Precision:", precision[i], "Recall:", recall[i])
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('P-R Curve')
            plt.savefig('PR.png')

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            print(classification_report(y_true, y_pred))
            print('Test accuracy:', accuracy)
        
            history.loss_plot('epoch')         

if __name__=='__main__':
    main()
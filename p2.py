# Load pickled data
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten, xavier_initializer_conv2d
import cv2

def loadData(dir):
    training_file = 'data/train.p'
    validation_file= 'data/valid.p'
    testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    return train, valid, test


### Data exploration visualization code goes here.
#def showInfo(data):


### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
def model(x, n_classes, keep_prob):
    mu  = 0
    sigma = 0.1

    depth = {
        'D_1': 16,
        'D_2': 32,
        'D_3': 512,
        'D_4': 8192,      
        'D_5': 256,      
    }

    weights = {
    #     'W_conv1':tf.Variable(tf.truncated_normal(shape=[5, 5, 1,            depth["D_1"]],  mean = mu, stddev = sigma, name = 'weight1')),
    #     'W_conv2':tf.Variable(tf.truncated_normal(shape=[5, 5, depth["D_1"], depth["D_2"]],  mean = mu, stddev = sigma, name = 'weight2')),
    #     'W_conv3':tf.Variable(tf.truncated_normal(shape=[5, 5, depth["D_2"], depth["D_3"]],  mean = mu, stddev = sigma, name = 'weight3')),

    #     'W_fc1':  tf.Variable(tf.truncated_normal(shape=[8192, depth["D_4"]], mean = mu, stddev = sigma, name = 'weight4')),
    #     'W_fc2':  tf.Variable(tf.truncated_normal(shape=[depth["D_4"],     depth["D_5"]], mean = mu, stddev = sigma, name = 'weight5')),
    #     'W_out':  tf.Variable(tf.truncated_normal(shape=[depth["D_5"],     n_classes],    mean = mu, stddev = sigma, name = 'weight_out')),

        'W_conv1':tf.Variable(xavier_initializer_conv2d()([5, 5, 1,            depth["D_1"]]),  name = 'weight1'),
        'W_conv2':tf.Variable(xavier_initializer_conv2d()([5, 5, depth["D_1"], depth["D_2"]]),  name = 'weight2'),
        'W_conv3':tf.Variable(xavier_initializer_conv2d()([5, 5, depth["D_2"], depth["D_3"]]),  name = 'weight3'),

        'W_fc1':  tf.Variable(xavier_initializer_conv2d()([8192,         depth["D_4"]]), name = 'weight4'),
        'W_fc2':  tf.Variable(xavier_initializer_conv2d()([depth["D_4"], depth["D_5"]]), name = 'weight5'),
        'W_out':  tf.Variable(xavier_initializer_conv2d()([depth["D_5"], n_classes]),    name = 'weight_out'),
 
    }

    biases = {
        'B_1':   tf.Variable(tf.zeros(depth['D_1']), name='bias_1'),
        'B_2':   tf.Variable(tf.zeros(depth['D_2']), name='bias_2'),
        'B_3':   tf.Variable(tf.zeros(depth['D_3']), name='bias_3'),
        'B_4':   tf.Variable(tf.zeros(depth['D_4']), name='bias_4'),
        'B_5':   tf.Variable(tf.zeros(depth['D_5']), name='bias_5'),
        'B_out': tf.Variable(tf.zeros(n_classes),    name='bias_out')
    }

    # print(x)
    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x1.
    conv1 = tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME') + biases['B_1']
    conv1 = tf.nn.relu(conv1) # Activation.
    # Pooling. Input = 32x32x1. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 2: Convolutional. Output = 10x10x32.
    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME') + biases['B_2']
    conv2 = tf.nn.relu(conv2) # Activation.
    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 2: Convolutional. Output = 10x10x32.
    conv3 = tf.nn.conv2d(conv2, weights['W_conv3'], strides=[1, 1, 1, 1], padding='SAME') + biases['B_3']
    conv3 = tf.nn.relu(conv3) # Activation.
    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Flatten. Input = 5x5x32. Output = 800.
    fc0 = flatten(conv3)

    # print("fc0", fc0, "weights:", weights["W_fc1"])
    # Layer 3: Fully Connected. Input = 800. Output = 120.
    fc1 = tf.matmul(fc0, weights['W_fc1']) + biases['B_4']
    fc1 = tf.nn.relu(fc1) # Activation.

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.matmul(fc1, weights['W_fc2']) + biases['B_5']
    fc2 = tf.nn.relu(fc2) # Activation.

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.matmul(fc2, weights['W_out']) + biases['B_out']
    
    return logits

def model2(x, n_classes, keep_prob):
    mu  = 0
    sigma = 0.1

    depth = {
        'D_1': 16,
        'D_2': 32,
        'D_3': 512,
        'D_4': 128,      
    }

    weights = {
        'W_conv1':tf.Variable(xavier_initializer_conv2d()([5, 5, 1,            depth["D_1"]]),  name = 'weight1'),
        'W_conv2':tf.Variable(xavier_initializer_conv2d()([5, 5, depth["D_1"], depth["D_2"]]),  name = 'weight2'),

        'W_fc1':  tf.Variable(xavier_initializer_conv2d()([5*5*depth["D_2"], depth["D_3"]]), name = 'weight4'),
        'W_fc2':  tf.Variable(xavier_initializer_conv2d()([depth["D_3"],     depth["D_4"]]), name = 'weight5'),

        'W_out':  tf.Variable(xavier_initializer_conv2d()([depth["D_4"], n_classes]),    name = 'weight_out'),
    }

    biases = {
        'B_1':   tf.Variable(tf.zeros(depth['D_1']), name='bias_1'),
        'B_2':   tf.Variable(tf.zeros(depth['D_2']), name='bias_2'),
        'B_3':   tf.Variable(tf.zeros(depth['D_3']), name='bias_3'),
        'B_4':   tf.Variable(tf.zeros(depth['D_4']), name='bias_4'),
        'B_out': tf.Variable(tf.zeros(n_classes),    name='bias_out')
    }

    # print(x)
    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x1.
    conv1 = tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 1, 1, 1], padding='VALID') + biases['B_1']
    conv1 = tf.nn.relu(conv1) # Activation.
    # Pooling. Input = 32x32x1. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Output = 10x10x32.
    conv2 = tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 1, 1, 1], padding='VALID') + biases['B_2']
    conv2 = tf.nn.relu(conv2) # Activation.
    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 5x5x32. Output = 800.
    fc0 = flatten(conv2)

    # print("fc0", fc0, "weights:", weights["W_fc1"])
    # Layer 3: Fully Connected. Input = 800. Output = 120.
    fc1 = tf.matmul(fc0, weights['W_fc1']) + biases['B_3']
    fc1 = tf.nn.relu(fc1) # Activation.

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.matmul(fc1, weights['W_fc2']) + biases['B_4']
    fc2 = tf.nn.relu(fc2) # Activation.

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.matmul(fc2, weights['W_out']) + biases['B_out']
    
    return logits

def evaluate(logits, one_hot_y, X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Preprocess Functions
def resize(img):
     # make sure the image is 32x32
    scaleFactorY = 32. / img.shape[0]
    scaleFactorX = 32. / img.shape[1]
    return cv2.resize(img, None, fx=scaleFactorX, fy=scaleFactorY, interpolation=cv2.INTER_AREA)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rotate(img):
    plt.imshow(img, cmap='gray')
    w, h = img.shape[0:2]     
    center = (w/2, h/2) 
    # Rotate between -20 to 20 degrees
    angle = int(random.random() * 40.0) - 20
    rotM = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(img, rotM, (w, h))
    return img

def warp(img):
    w, h = img.shape[0:2]     
    x = int(random.random() * 4) - 2
    y = int(random.random() * 4) - 2
    transM = np.float32([[1, 0, x], [0, 1, y]])
    img = cv2.warpAffine(img, transM, (w, h))
    return img

def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    o_min = -1.
    o_max = 1.
    x_min = 0.
    x_max = 255.
    return o_min + (image_data - x_min) * (o_max - o_min) / (x_max - x_min)

def equalized(img):
     return cv2.equalizeHist(img).reshape(32,32,1)

def preprocess(xData, yData, shake):
    newX = np.empty([xData.shape[0], xData.shape[1], xData.shape[2], 1])

    for i, img in enumerate(xData):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if(shake):
            gray = warp(gray)
            gray = rotate(gray)
        equ = cv2.equalizeHist(gray).reshape(32,32,1)
        newX[i] = equ

    newX = normalize(newX)
    return newX, yData

def preprocess2(xData, yData):
    preprocess_steps = []
    
    # plt.imshow(xData[0])
    # plt.show()

    print("Initial:", xData[0].shape)
    
    # Resize
    preprocess_steps.append("Resized")
    for i in range(len(xData)):
        xData[i] = resize(xData[i])
    print("Resize:", xData[0].shape)

    # Grayscale
    preprocess_steps.append("Grayscaled")
    for i in range(len(xData)):
        xData[i] = grayscale(xData[i])
    print("Grayscale:", xData[0].shape)

    # Rotate
    # if (steps % (2**2) > 0): 
    #     preprocess_steps.append("Rotated")
    #     for i in range(len(xData)):
    #         xData[i] = rotate(xData[i])
    
    # Warp  
    preprocess_steps.append("Warped")
    for i in range(len(xData)):
        xData[i] = warp(xData[i])
    print("Warp:", xData[0].shape)

    # Normalize
    preprocess_steps.append("Normalized")
    xData = normalize(xData)
    print("Normalize:", xData[0].shape)

    print("Preprocess Steps Taken: ", preprocess_steps)
    return xData, yData
    
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.

### Load the images and plot them here.

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.

train, valid, test = loadData('data')

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train.shape
n_classes = np.max(train['labels'] + 1)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001
EPOCHS = 15
BATCH_SIZE = 128

logits = model2(x, n_classes, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()

    X_valid, y_valid = preprocess(X_valid, y_valid, False)

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        X_train_pp, y_train_pp = preprocess(X_train, y_train, False)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_pp[offset:end], y_train_pp[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})
            
        validation_accuracy = evaluate(logits, one_hot_y, X_valid, y_valid)
        print("EPOCH {:2d} - Acc = {:4f}".format(i+1, validation_accuracy))
    print()
        
    saver.save(sess, './model')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    X_test, y_test = preprocess(X_test, y_test, False)
    test_accuracy = evaluate(logits, one_hot_y, X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

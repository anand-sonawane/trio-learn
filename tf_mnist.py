import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#lets define functions for the above layers

#construct a convolutional layer
def convolution_layer(layer_name, input_maps, num_output_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
    """
    The input image shape is 224 x 224 x 3
    kernel size for vgg16 is defined as [3,3]
    stride of 1 is going to be used
    """
    num_input_channels = input_maps.get_shape()[-1].value
    
    with tf.name_scope(layer_name) as scope:
        
        kernel = tf.get_variable(scope+'W',
                                 shape=[kernel_size[0], kernel_size[1], num_input_channels, num_output_channels],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        
        convolution = tf.nn.conv2d(input_maps, kernel, stride, padding='SAME')
        
        bias = tf.Variable(tf.constant(0.0, shape=[num_output_channels], dtype=tf.float32), trainable=True, name='b')
        
        output = tf.nn.relu(tf.nn.bias_add(convolution, bias), name=scope)
        
    return output, kernel, bias

# construct a max pooling layer
def max_pooling_layer(layer_name, input_maps, kernel_size=[2, 2], stride=[1, 2, 2, 1]):
    output = tf.nn.max_pool(input_maps,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=stride,
                            padding='SAME',
                            name=layer_name)
    return output


# construct a average pooling layer
def avg_pooling_layer(layer_name, input_maps, kernel_size=[2, 2], stride=[1, 2, 2, 1]):
    output = tf.nn.avg_pool(input_maps,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=stride,
                            padding='SAME',
                            name=layer_name)
    return output


# construct a fully connection layer
def fully_connection_layer(layer_name, input_maps, num_output_nodes,activation):
    shape = input_maps.get_shape()
    
    #if layer == 4 then we will flatten the layer
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    with tf.name_scope(layer_name) as scope:
        kernel = tf.get_variable(scope+'W',
                                 shape=[size, num_output_nodes],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[num_output_nodes], dtype=tf.float32), trainable=True, name='b')
        flat = tf.reshape(input_maps, [-1, size])
        if(activation == "relu"):
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, kernel), bias))
        else:
            output = tf.nn.softmax(tf.nn.bias_add(tf.matmul(flat, kernel), bias))
    return output, kernel, bias

#lets create the vgg model
def vgg16(input_maps, num_classes=1000, isTrain=False, keep_prob=1.0):
    
    # assume the input image shape is 224 x 224 x 3
    
    # Two convolutional layers : 224 x 224 x 3 to 224 x 224 x 64

    output1_1, kernel1_1, bias1_1 = convolution_layer('conv1_1', input_maps, 64)
    output1_2, kernel1_2, bias1_2 = convolution_layer('conv1_2', output1_1, 64)
    
    # 224 x 224 x 64 to output1_3 shape 112 x 112 x 64
    
    output1_3 = max_pooling_layer('pool1', output1_2)
    
    # Two convolutional layers : 112 x 112 x 64 to 112 x 112 x 128

    output2_1, kernel2_1, bias2_1 = convolution_layer('conv2_1', output1_3, 128)
    output2_2, kernel2_2, bias2_2 = convolution_layer('conv2_2', output2_1, 128)
    
    # 112 x 112 x 128 to output2_3 shape 56 x 56 x 128
    
    output2_3 = max_pooling_layer('pool2', output2_2)
    
    # Three convolutional layers : 56 x 56 x 128 to 56 x 56 x 256
    
    output3_1, kernel3_1, bias3_1 = convolution_layer('conv3_1', output2_3, 256)
    output3_2, kernel3_2, bias3_2 = convolution_layer('conv3_2', output3_1, 256)
    output3_3, kernel3_3, bias3_3 = convolution_layer('conv3_3', output3_2, 256)
    
    # 56 x 56 x 256 to output3_4 shape 28 x 28 x 256
    
    output3_4 = max_pooling_layer('pool3', output3_3)
    
    # Three convolutional layers : 28 x 28 x 256 to 28 x 28 x 512
    
    output4_1, kernel4_1, bias4_1 = convolution_layer('conv4_1', output3_4, 512)
    output4_2, kernel4_2, bias4_2 = convolution_layer('conv4_2', output4_1, 512)
    output4_3, kernel4_3, bias4_3 = convolution_layer('conv4_3', output4_2, 512)
    
    # 28 x 28 x 512 to output3_4 shape 14 x 14 x 512
    
    output4_4 = max_pooling_layer('pool4', output4_3)

    # Three convolutional layers : 14 x 14 x 512 to 14 x 14 x 512

    output5_1, kernel5_1, bias5_1 = convolution_layer('conv5_1', output4_4, 512)
    output5_2, kernel5_2, bias5_2 = convolution_layer('conv5_2', output5_1, 512)
    output5_3, kernel5_3, bias5_3 = convolution_layer('conv5_3', output5_2, 512)
    
    # 14 x 14 x 512 to output3_4 shape 7 x 7 x 512
    
    output5_4 = max_pooling_layer('pool5', output5_3)
    
    # output5_4 shape 7 x 7 x 512 flattened to 1 x 4096

    output6_1, kernel6_1, bias6_1 = fully_connection_layer('fc6_1', output5_4, 4096, "relu")
    
    # fc6_2 shape 1 x 4096 flattened to 1 x 4096
    
    output6_2, kernel6_2, bias6_2 = fully_connection_layer('fc6_2', output6_1, 4096, "relu")
    
    # fc6_2 shape 1 x 4096 flattened to num_classes for softmax
    
    output6_3, kernel6_3, bias6_3 = fully_connection_layer('fc6_3', output6_2, num_classes, "softmax")
    
    return output6_3

def training(train_x, train_y, valid_x=None, valid_y=None, format_size=[224, 224],
             batch_size=10, learn_rate=0.01, num_epochs=1, save_model=False, debug=False):
    """
    Function for training the VGG Model
    @params:
    
    train_x is a 4-D matrix : [num_images, img_height, img_width, num_channels]
    train_y is a 2-D matrix : [num_images, num_classes] (using one-hot labels)
    valid_x is a 4-D matrix like train_x
    valid_y is a 2-D matrix like train_y
    format size : Input image size
    batch size : batch size
    learn rate : model learning rate
    num_epochs : Number of Epochs to perform model training
    save_model = Do you want to save the model or not ?
    debug = Do you want to debug or not ?
    
    """
    
    #assert len(train_x.shape) == 4
    [num_images, img_height, img_width, num_channels] = train_x.shape
    num_classes = train_y.shape[-1]
    num_steps = int(np.ceil(num_images / float(batch_size)))

    # build the graph and define objective function
    graph = tf.Graph()
    with graph.as_default():
        
        # build graph
        train_maps_raw = tf.placeholder(tf.float32, [None, img_height, img_width, num_channels])
        train_maps = tf.image.resize_images(train_maps_raw, [format_size[0], format_size[1]])
        train_labels = tf.placeholder(tf.float32, [None, num_classes])
        logits = vgg16(train_maps, num_classes, isTrain=True, keep_prob=0.6)

        # loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = train_labels)
        loss = tf.reduce_mean(cross_entropy)

        # optimizer with decayed learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps*num_epochs, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # prediction for the training data
        train_prediction = tf.nn.softmax(logits)
        
    print("graph ready")
    
    # train the graph
    with tf.Session(graph=graph) as session:
        # saver to save the trained model
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            print(epoch)
            for step in range(num_steps):
                print(step)
                offset = (step * batch_size) % (num_images - batch_size)
                batch_data = train_x[offset:(offset + batch_size), :, :, :]
                batch_labels = train_y[offset:(offset + batch_size), :]
                feed_dict = {train_maps_raw: batch_data, train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                if debug:
                    if step % int(np.ceil(num_steps/2.0)) == 0:
                        print('Epoch %2d/%2d step %2d/%2d: ' % (epoch+1, num_epochs, step, num_steps))
                        print('\tBatch Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, batch_labels)))
                        if valid_x is not None:
                            feed_dict = {train_maps_raw: valid_x, train_labels: valid_y}
                            l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                            print('\tValid Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, valid_y)))

            print ('Epoch %2d/%2d:\n\tTrain Loss = %.2f\t Accuracy = %.2f%%' %
                   (epoch+1, num_epochs, l, accuracy(predictions, batch_labels)))
            if valid_x is not None and valid_y is not None:
                feed_dict = {train_maps_raw: valid_x, train_labels: valid_y}
                l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                print('\tValid Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, valid_y)))

            # Save the variables to disk
            if save_model:
                #saver.save(session, 'my_test_model',global_step=1000)
                #save_path = saver.save(session, 'model')
                #print('The model has been saved to ' + save_path)

                #tf.train.write_graph(session.graph,'/home/aodev/trio','saved_model.pb', as_text=False)
                tf.train.write_graph(session.graph,'./weights/','saved_model'+str(epoch)+'.pb', as_text=False)

        session.close()


# predictions is a 2-D matrix [num_images, num_classes]
# labels is a 2-D matrix like predictions
def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def get_data():
    #load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    #read data
    train_features = mnist.train.images
    train_labels = mnist.train.labels

    test_features = mnist.test.images
    test_labels = mnist.test.labels
    
    return train_features,train_labels,test_features,test_labels

train_x, train_y, valid_x, valid_y = get_data()
train_x_less = train_x[0:100,:]
train_y_less = train_y[0:100, :]

#hyperparameters
format_size = [224, 224]
batch_size=10
learn_rate=0.01 
num_epochs=100

train_x1 = train_x_less.reshape(100,28,28,1)
valid_x1 = valid_x.reshape(10000,28,28,1)

training(train_x1, train_y_less, None, None,format_size,batch_size,learn_rate,num_epochs,save_model=True, debug=True)
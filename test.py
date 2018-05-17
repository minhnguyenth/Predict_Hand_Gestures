"""
Code to calculate the average accuracy rate when applying the trained model on testing data is stored here
"""

import prepdata
import tensorflow as tf
import numpy as np

# Prepare input data
classes = ['A', 'B', 'C', 'Five', 'Point', 'V']
num_classes = len(classes)

img_size = 100
num_channels = 3
train_path = 'Marcel-Test'

testing_data = prepdata.load_train(train_path, img_size, classes)



x_batch = testing_data[0]
## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('hand_gestures_model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

y_test_images = np.zeros((len(testing_data[0]), 6))


### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
y_predicted = tf.argmax(result, dimension=1)
y_true = tf.argmax(testing_data[1], dimension=1)
correct_prediction = tf.equal(y_predicted, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = sess.run(accuracy)
print(acc)
temp = 0
import tensorflow

def max_out_loss(distance):
    distance_plus1 = tf.add(distance, 1)
    maxed_distance_plus1 = tf.nn.relu(distance_plus1)
    maxed_distance = tf.subtract(distance, 1)


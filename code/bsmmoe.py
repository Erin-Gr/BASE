from bslayer import mmoe_layer, tower_layer
import tensorflow as tf
from tensorflow.keras import Model


class TBRE(Model):
    def __init__(self, maxlen, mmoe_hidden_units, num_experts_sh, num_experts_sp, num_tasks,
                 tower_hidden_units, output_dim, behavior_id, dropout_rate, l2_reg, is_training, activation='relu',
                 use_expert_bias=True, use_gate_bias=True, **kwargs):
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        super(TBRE, self).__init__()
        self.mmoe_layer = mmoe_layer(mmoe_hidden_units,
                                     num_experts_sh,
                                     num_experts_sp,
                                     num_tasks,
                                     behavior_id,
                                     dropout_rate,
                                     l2_reg,
                                     is_training,
                                     use_expert_bias,
                                     use_gate_bias)
 
        self.tower_layer = [
            tower_layer(tower_hidden_units, output_dim, activation)
            for _ in range(num_tasks)
        ]
        self.num_tasks = num_tasks
        self.output_dim =output_dim
        self.maxlen = maxlen
        self.behavior_id = behavior_id
        self.num_units = output_dim
        
        self.purchase_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 0)), -1)  # mask purchase
        self.cart_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 1)), -1)  # mask cart
        self.favorite_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 2)), -1)  # mask favorite
        self.view_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 3)), -1)  # mask view
 
    def call(self, inputs, target_inputs):
        mmoe_outputs = self.mmoe_layer(inputs, target_inputs)  
        outputs = [] 
        for i, layer in enumerate(self.tower_layer):
            out = layer(mmoe_outputs[i], target_inputs)
            outputs.append(out) # list: num_tasks × [bs,L,d]
            
        for i in range(self.num_tasks):
            outputs[i] = self.layer_normalization(outputs[i]) #+ inputs
            outputs[i] = tf.layers.dropout(outputs[i], rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            
        return outputs[0], outputs[1], outputs[2]
        
    def layer_normalization(self, inputs, epsilon=1e-8):
        with tf.compat.v1.variable_scope("layer_normalization"):
            alpha = tf.Variable(tf.ones(self.num_units))
            beta = tf.Variable(tf.zeros(self.num_units))

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = alpha * normalized + beta

        return outputs

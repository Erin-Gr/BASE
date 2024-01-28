import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from loguru import logger

class mmoe_layer(Layer):
    def __init__(self, hidden_units, num_expert_sh, num_expert_sp, num_tasks, behavior_id,
                 dropout_rate, l2_reg, is_training, use_expert_bias=True, use_gate_bias=True, **kwargs):

        super(mmoe_layer, self).__init__()
        self.hidden_units = hidden_units
        self.num_expert_sh = num_expert_sh
        self.num_expert_sp = num_expert_sp 
        self.num_tasks = num_tasks
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.is_training = is_training
        self.use_expert_bias = use_expert_bias
        self.purchase_use_expert_bias = use_expert_bias
        self.cart_use_expert_bias = use_expert_bias
        self.favorite_use_expert_bias = use_expert_bias
        self.view_use_expert_bias = use_expert_bias

        self.use_gate_bias = use_gate_bias

        self.behavior_id = behavior_id
        self.purchase_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 0)), -1)  # mask purchase
        self.cart_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 1)), -1)  # mask cart
        self.favorite_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 2)), -1)  # mask favorite
        self.view_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 3)), -1)  # mask view

    #
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("The dimension of inputs should be 3, not %d" % (len(input_shape)))
        
        self.expert_matrix = tf.Variable(tf.truncated_normal(shape=[input_shape[-1], self.hidden_units, self.num_expert_sh],
                                                             mean=0.0,
                                                             stddev=tf.sqrt(tf.div(2.0, 3 * self.hidden_units + 1))),
                                         name='expert_matrix',
                                         dtype=tf.float32)
        
        
        self.purchase_expert_matrix = tf.Variable(tf.truncated_normal(shape=[input_shape[-1], self.hidden_units, self.num_expert_sp],
                                                             mean=0.0,
                                                             stddev=tf.sqrt(tf.div(2.0, 3 * self.hidden_units + 1))),
                                         name='purchase_expert_matrix',
                                         dtype=tf.float32)
        
        
        self.favorite_expert_matrix = tf.Variable(tf.truncated_normal(shape=[input_shape[-1], self.hidden_units, self.num_expert_sp],
                                                             mean=0.0,
                                                             stddev=tf.sqrt(tf.div(2.0, 3 * self.hidden_units + 1))),
                                         name='favorite_expert_matrix',
                                         dtype=tf.float32)
        
        self.view_expert_matrix = tf.Variable(tf.truncated_normal(shape=[input_shape[-1], self.hidden_units, self.num_expert_sp],
                                                             mean=0.0,
                                                             stddev=tf.sqrt(tf.div(2.0, 3 * self.hidden_units + 1))),
                                         name='view_expert_matrix',
                                         dtype=tf.float32)
        
        
        self.gate_matrix = [tf.Variable(tf.truncated_normal(shape=[input_shape[-1], self.num_expert_sh+self.num_expert_sp],
                                                             mean=0.0,
                                                             stddev=tf.sqrt(tf.div(2.0, 3 * self.hidden_units + 1))),
                                         name='gate_matrix'+str(i),
                                         dtype=tf.float32) for i in range(self.num_tasks)]
        
        if self.use_expert_bias:
            self.expert_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_units, self.num_expert_sh]),
                                           name='expert_bias',
                                           dtype=tf.float32)

        if self.purchase_use_expert_bias:
            self.purchase_expert_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_units, self.num_expert_sp]),
                                           name='purchase_expert_bias',
                                           dtype=tf.float32)
        
        if self.favorite_use_expert_bias:
            self.favorite_expert_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_units, self.num_expert_sp]),
                                           name='favorite_expert_bias',
                                           dtype=tf.float32)
        
        if self.view_use_expert_bias:
            self.view_expert_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_units, self.num_expert_sp]),
                                           name='view_expert_bias',
                                           dtype=tf.float32)
            
        if self.use_gate_bias:
            self.gate_bias = [tf.Variable(tf.constant(0.0, shape=[self.num_expert_sh + self.num_expert_sp]),
                                           name='gate_bias'+str(i),
                                           dtype=tf.float32) for i in range(self.num_tasks)]
            
    def call(self, inputs_res, target_inputs, **kwargs):
        if K.ndim(inputs_res) != 3:
            raise ValueError("The dim of inputs should be 3, not %d" % (K.ndim(inputs)))
        
        inputs = inputs_res

        expert_output = []
        for i in range(self.num_expert_sh):
            expert_out = tf.matmul(inputs, tf.expand_dims(self.expert_matrix[:, :, i], axis=0)) 
            expert_output.append(expert_out) 
        expert_output = tf.transpose(tf.convert_to_tensor(expert_output),
                                     [1, 2, 3, 0]) 

        purchase_inputs = inputs
        purchase_expert_output = []
        for i in range(self.num_expert_sp):
            purchase_expert_out = tf.matmul(purchase_inputs, tf.expand_dims(self.purchase_expert_matrix[:, :, i], axis=0))  
            purchase_expert_output.append(purchase_expert_out)
        purchase_expert_output = tf.transpose(tf.convert_to_tensor(purchase_expert_output),
                                     [1, 2, 3, 0])

        favorite_inputs = inputs
        favorite_expert_output = []
        for i in range(self.num_expert_sp):
            favorite_expert_out = tf.matmul(favorite_inputs, tf.expand_dims(self.favorite_expert_matrix[:, :, i], axis=0)) 
            favorite_expert_output.append(favorite_expert_out) 
        favorite_expert_output = tf.transpose(tf.convert_to_tensor(favorite_expert_output),
                                     [1, 2, 3, 0]) 

        view_inputs = inputs
        view_expert_output = []
        for i in range(self.num_expert_sp):
            view_expert_out = tf.matmul(view_inputs, tf.expand_dims(self.view_expert_matrix[:, :, i], axis=0)) 
            view_expert_output.append(view_expert_out) 
        view_expert_output = tf.transpose(tf.convert_to_tensor(view_expert_output),
                                     [1, 2, 3, 0])
     
        if self.use_expert_bias:
            expert_output += self.expert_bias
        expert_output = tf.nn.relu(expert_output) # [bs, None, hidden_units, num_expert_sh]
        if self.purchase_use_expert_bias:
            purchase_expert_output += self.purchase_expert_bias
        purchase_expert_output = tf.nn.relu(purchase_expert_output) # [bs, None, hidden_units, num_expert_sp]
        if self.favorite_use_expert_bias:
            favorite_expert_output += self.favorite_expert_bias
        favorite_expert_output = tf.nn.relu(favorite_expert_output)
        if self.view_use_expert_bias:
            view_expert_output += self.view_expert_bias
        view_expert_output = tf.nn.relu(view_expert_output)
        
        gate_outputs = []
        for i, gate in enumerate(self.gate_matrix):
            gate_out = tf.matmul(inputs, tf.expand_dims(gate,axis=0))
            if self.use_gate_bias:
                gate_out += self.gate_bias[i]
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.layers.dropout(gate_out, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))

            gate_outputs.append(gate_out)  # list: num_tasks x [bs, None, num_expert_sh+num_expert_sp]
   
        outputs = []
        for gate_index, gate_out in enumerate(gate_outputs):
            gate_out = tf.expand_dims(gate_out, axis=2) 
            gate_out = tf.tile(gate_out, [1, 1, self.hidden_units, 1]) 
            if gate_index == 0:
                purchase_output = tf.concat([expert_output, purchase_expert_output], axis=-1)
                out = tf.multiply(gate_out, purchase_output)
            if gate_index == 1:
                favorite_output = tf.concat([expert_output, favorite_expert_output], axis=-1)
                out = tf.multiply(gate_out, favorite_output)
            if gate_index == 2:
                view_output = tf.concat([expert_output, view_expert_output], axis=-1)
                out = tf.multiply(gate_out, view_output)

            out = tf.reduce_sum(out, axis=-1)
            outputs.append(out)
        return outputs  # list: num_tasks x [bs, None, hidden_units]

class tower_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation='relu'):
        super(tower_layer, self).__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in [hidden_units]]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs, target_behavior_emb, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("The dim of inputs should be 3, not %d" % (K.ndim(inputs)))
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.array_ops import boolean_mask
from tensorflow.python.ops.array_ops import sequence_mask
from tensorflow.python.ops.array_ops import _all_dimensions
import tensorflow as tf
from bsmmoe import TBRE
from tensorflow.keras.layers import Dense

class Model(object):
    def __init__(self, num_user, num_item, num_cat, num_seller, num_brand, args):
        # ==== some configurations ====
        self.num_user = num_user
        self.num_item = num_item
        self.num_cat = num_cat
        self.num_seller = num_seller # if kaggle, then num_seller=0
        self.num_brand = num_brand
        self.maxlen = args.maxlen
        self.num_units = args.hidden_units
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.l2_reg = args.l2_reg
        self.num_behavior = args.num_behavior
        self.ext_modules = args.ext_modules
        self.dataset = args.dataset
        self.num_expert_sh = args.num_expert_sh
        self.num_expert_sp = args.num_expert_sp
        self.num_tasks = 3
        self.mmoe_hidden_units = args.mmoe_hidden_units
        self.tower_hidden_units = args.tower_hidden_units
        self.batch_size = args.batch_size

        # # ==== input batch data ====
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.user_id = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.input_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.pos_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.neg_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.behavior_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.target_behavior = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        
        self.cat_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.target_cat = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.seller_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.target_seller = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.brand_id = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))
        self.target_brand = tf.compat.v1.placeholder(tf.int32, shape=(None, self.maxlen))

        padding_mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_id, 0)), -1)
        self.purchase_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 0)), -1)  # mask purchase
        self.favorite_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 2)), -1)  # mask favorite for Tmall; mask cart for kaggle
        self.view_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 3)), -1)  # mask view

        # ==== item embedding matrix ====
        with tf.compat.v1.variable_scope("item_embeddings"):
            # items are indexed from 1 so that the max row is num_item+1
            self.item_emb_mat = tf.compat.v1.get_variable('lookup_table',
                                                          dtype=tf.float32,
                                                          shape=[self.num_item, self.num_units],
                                                          regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.item_emb_mat = tf.concat(
                (tf.zeros(shape=[1, self.num_units]), self.item_emb_mat), 0) * (self.num_units ** 0.5)

            item_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.input_id)

        # ==== attribute embedding matrix ====
        with tf.compat.v1.variable_scope("cat_embeddings"):
            # items are indexed from 1 so that the max row is num_item+1
            self.cat_emb_mat = tf.compat.v1.get_variable('lookup_table',
                                                          dtype=tf.float32,
                                                          shape=[self.num_cat, self.num_units],
                                                          regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.cat_emb_mat = tf.concat(
                (tf.zeros(shape=[1, self.num_units]), self.cat_emb_mat), 0) * (self.num_units ** 0.5)

            cat_embs = tf.nn.embedding_lookup(self.cat_emb_mat, self.cat_id)    #[bs, L ,d]
          
        if self.dataset == 'Tmall':
            with tf.compat.v1.variable_scope("seller_embeddings"):
                # items are indexed from 1 so that the max row is num_item+1
                self.seller_emb_mat = tf.compat.v1.get_variable('lookup_table',
                                                            dtype=tf.float32,
                                                            shape=[self.num_seller, self.num_units],
                                                            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
                self.seller_emb_mat = tf.concat(
                    (tf.zeros(shape=[1, self.num_units]), self.seller_emb_mat), 0) * (self.num_units ** 0.5)

                seller_embs = tf.nn.embedding_lookup(self.seller_emb_mat, self.seller_id)    #[bs, L ,d]
              
        with tf.compat.v1.variable_scope("brand_embeddings"):
            # items are indexed from 1 so that the max row is num_item+1
            self.brand_emb_mat = tf.compat.v1.get_variable('lookup_table',
                                                          dtype=tf.float32,
                                                          shape=[self.num_brand, self.num_units],
                                                          regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.brand_emb_mat = tf.concat(
                (tf.zeros(shape=[1, self.num_units]), self.brand_emb_mat), 0) * (self.num_units ** 0.5)

            brand_embs = tf.nn.embedding_lookup(self.brand_emb_mat, self.brand_id)    #[bs, L ,d]
            
        # ==== position embedding matrix ====
        with tf.compat.v1.variable_scope("position_embeddings"):
            pos_emb_mat = tf.compat.v1.get_variable('lookup_table',
                                                    dtype=tf.float32,
                                                    shape=[self.maxlen, self.num_units],
                                                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))

            pos_embs = tf.nn.embedding_lookup(pos_emb_mat, tf.range(self.maxlen))
        
        # ==== behavior embedding matrix ====
        with tf.variable_scope("behavior_embeddings"):
            self.beha_emb_mat = tf.get_variable('lookup_table',
                                                dtype=tf.float32,
                                                shape=[args.num_behavior + 1, self.num_units],
                                                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            target_beha_emb = tf.nn.embedding_lookup(self.beha_emb_mat, self.target_behavior)  # [bs, L, d]
            beha_embs = tf.nn.embedding_lookup(self.beha_emb_mat, self.behavior_id) # [bs, L, d]
            
        # ==== input embedding matrix ====
        with tf.compat.v1.variable_scope("input_matrix"):
            inputs = item_embs + pos_embs
            inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            inputs = inputs * padding_mask
            inputs = self.layer_normalization(inputs)

            inputs_cat = cat_embs + pos_embs
            inputs_cat = tf.layers.dropout(inputs_cat, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            inputs_cat = inputs_cat * padding_mask
            inputs_cat = self.layer_normalization(inputs_cat)
            
            if self.dataset == 'Tmall':
                inputs_seller = seller_embs + pos_embs
                inputs_seller = tf.layers.dropout(inputs_seller, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
                inputs_seller = inputs_seller * padding_mask
                inputs_seller = self.layer_normalization(inputs_seller)
            
            inputs_brand = brand_embs + pos_embs
            inputs_brand = tf.layers.dropout(inputs_brand, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            inputs_brand = inputs_brand * padding_mask
            inputs_brand = self.layer_normalization(inputs_brand)

        # ==== item sequence output initialization ====
        loc_embs = tf.zeros([tf.shape(inputs)[0], self.maxlen, self.num_units])

        # ==== local representation (hierarchical self-attention blocks) ====
        # if 'l' in args.ext_modules:
        loc_embs = inputs
        for b in range(args.num_blocks):
            with tf.compat.v1.variable_scope("self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs, laststep_attention = self.multihead_attention(loc_embs,
                                                                        padding_mask=padding_mask,
                                                                        num_heads=args.num_heads,
                                                                        dropout_rate=args.dropout_rate,
                                                                        score='scaled_dot',
                                                                        causality=True,
                                                                        get_att='lastq_ave',
                                                                        residual=True)
                loc_embs = self.layer_normalization(loc_embs)
                # feed forward
                loc_embs = self.feed_forward(loc_embs)
                loc_embs *= padding_mask
                loc_embs = self.layer_normalization(loc_embs)
                if b == 0:
                    self.bottom_laststep_attention = laststep_attention
                if b == args.num_blocks - 1:
                    self.top_laststep_attention = laststep_attention
            
        attention_mask_b = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='b')

        loc_embs_b = inputs * self.purchase_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("buy-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_b, laststep_attention_b = self.multihead_attention_ub(loc_embs_b,
                                                                               loc_embs,
                                                                               attention_mask=attention_mask_b,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_b = self.layer_normalization(loc_embs_b)
                loc_embs_b = self.feed_forward(loc_embs_b)
                loc_embs_b *= self.purchase_mask
                loc_embs_b = self.layer_normalization(loc_embs_b)
                loc_embs_b *= self.purchase_mask

        attention_mask_f = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='f')

        loc_embs_f = inputs * self.favorite_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("fav-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_f, laststep_attention_f = self.multihead_attention_ub(loc_embs_f,
                                                                               loc_embs,
                                                                               attention_mask=attention_mask_f,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_f = self.layer_normalization(loc_embs_f)
                loc_embs_f = self.feed_forward(loc_embs_f)
                loc_embs_f *= self.favorite_mask
                loc_embs_f = self.layer_normalization(loc_embs_f)
                loc_embs_f *= self.favorite_mask

        attention_mask_v = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='v')
        
        loc_embs_v = inputs * self.view_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("view-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_v, laststep_attention_v = self.multihead_attention_ub(loc_embs_v,
                                                                               loc_embs,
                                                                               attention_mask=attention_mask_v,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_v = self.layer_normalization(loc_embs_v)
                loc_embs_v = self.feed_forward(loc_embs_v)
                loc_embs_v *= self.view_mask
                loc_embs_v = self.layer_normalization(loc_embs_v)
                loc_embs_v *= self.view_mask
        
        if 'a' in self.ext_modules:
            with tf.variable_scope("cat1_attention"):
                loc_cat, self.attention_cat_lastq_ave, self.attention_cat_lastq_multi, self.attention_cat_batch_multi = self.multihead_attention_cat(query=target_beha_emb, inputs=inputs_cat, padding_mask=padding_mask, dropout_rate=args.dropout_rate)
                loc_cat = self.layer_normalization(loc_cat)
                loc_cat = self.feed_forward(loc_cat)
                loc_cat *= padding_mask
                loc_cat = self.layer_normalization(loc_cat)
            
            if self.dataset == 'Tmall' or self.dataset == 'JD':
                with tf.variable_scope("seller1_attention"):
                    loc_seller,self.attention_seller_lastq_ave, self.attention_seller_lastq_multi, self.attention_seller_batch_multi = self.multihead_attention_bran(query=target_beha_emb, inputs=inputs_seller, padding_mask=padding_mask, dropout_rate=args.dropout_rate)
                    loc_seller = self.layer_normalization(loc_seller)
                    loc_seller = self.feed_forward(loc_seller)
                    loc_seller *= padding_mask
                    loc_seller = self.layer_normalization(loc_seller)
            with tf.variable_scope("brand1_attention"):
                loc_brand, self.attention_brand_lastq_ave, self.attention_brand_lastq_multi, self.attention_brand_batch_multi = self.multihead_attention_sell(query=target_beha_emb, inputs=inputs_brand, padding_mask=padding_mask, dropout_rate=args.dropout_rate)
                loc_brand = self.layer_normalization(loc_brand)
                loc_brand = self.feed_forward(loc_brand)
                loc_brand *= padding_mask
                loc_brand = self.layer_normalization(loc_brand)
                
            if self.dataset == 'Tmall' or self.dataset == 'JD':
                stac = tf.concat([loc_cat, loc_seller, loc_brand], axis=-1)
                stac = tf.reshape(stac, [-1, self.maxlen, 3, self.num_units])
            else:
                stac = tf.concat([loc_cat, loc_brand], axis=-1)
                stac = tf.reshape(stac, [-1, self.maxlen, 2, self.num_units])
                
            loc_context = self.dynamic_att_attention(query=beha_embs, inputs=stac, dropout_rate=args.dropout_rate)
            loc_context *= padding_mask
            loc_context_test = tf.expand_dims(loc_context[:, -1, :], 1)
        
        # ==== hybrid/final respresentations ====
        positive_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.pos_id)
        negative_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.neg_id)
        
        if 'b' in self.ext_modules:
            loc_embs_multiview = loc_embs_b + loc_embs_f + loc_embs_v 
            context_b, context_f, context_v = self.get_ubp_memory(loc_embs_multiview)
            with tf.variable_scope("gating_parameters"):
                stacked = tf.concat([context_b, context_f, context_v], axis=-1)
                stacked = tf.reshape(stacked, [-1, self.maxlen, 3 * self.num_units])  # [bs, L, 3d]      
            mmoe_model = TBRE(self.maxlen, self.mmoe_hidden_units, self.num_expert_sh, self.num_expert_sp, self.num_tasks,
                            self.tower_hidden_units, self.num_units, self.behavior_id, self.dropout_rate, self.l2_reg,
                            self.is_training)
            mmoe_outputs_b, mmoe_outputs_f, mmoe_outputs_v = mmoe_model.call(stacked, target_beha_emb)
            mmoe_outputs = self.get_target_view(mmoe_outputs_b, mmoe_outputs_f, mmoe_outputs_v)
        
        outputs = loc_embs
        outputs_test = loc_embs

        if 'a' in self.ext_modules:
            outputs += loc_context
            outputs_test += loc_context_test

        if 'b' in self.ext_modules:
            outputs += mmoe_outputs
            outputs_test += mmoe_outputs_b
    
        outputs_te = tf.expand_dims(outputs_test[:, -1, :], 1)
        if 't' in self.ext_modules:
            with tf.variable_scope("attributes_contribution_gating"):
                if self.dataset == 'Tmall':
                    gatings = tf.concat([cat_embs, seller_embs, brand_embs, loc_embs], axis=-1)  # [bs, L, 4d]
                    gatings = tf.nn.softmax(tf.layers.dense(gatings, 3, activation=None))
                    cat_contri = tf.expand_dims(gatings[:, :, 0], -1)
                    seller_contri = tf.expand_dims(gatings[:, :, 1], -1)
                    brand_contri = tf.expand_dims(gatings[:, :, 2], -1)
                else:
                    gatings = tf.concat([cat_embs, brand_embs, loc_embs], axis=-1) 
                    gatings = tf.nn.softmax(tf.layers.dense(gatings, 2, activation=None))
                    cat_contri = tf.expand_dims(gatings[:, :, 0], -1) 
                    brand_contri = tf.expand_dims(gatings[:, :, 1], -1)
            
            self.auxiliary_mask = self.favorite_mask + self.view_mask #[bs, L, d]
            
            if self.dataset == 'Tmall':
                anchor_b = loc_embs * self.purchase_mask
                anchor_aux = loc_embs * self.auxiliary_mask
                pos_cat_b = cat_embs * self.purchase_mask  
                pos_seller_b = seller_embs * self.purchase_mask
                pos_brand_b = brand_embs * self.purchase_mask
                anchor_all = tf.concat([anchor_b, anchor_b, anchor_b, anchor_aux], axis=1)
                pos_att_aux = cat_embs * cat_contri + seller_embs * seller_contri + brand_embs * brand_contri 
                pos_att_aux *= self.auxiliary_mask
                pos_all = tf.concat([pos_cat_b, pos_seller_b, pos_brand_b, pos_att_aux], axis=1)
            else:
                anchor_b = loc_embs * self.purchase_mask
                anchor_aux = loc_embs * self.auxiliary_mask
                pos_cat_b = cat_embs * self.purchase_mask
                pos_brand_b = brand_embs * self.purchase_mask
                anchor_all = tf.concat([anchor_b, anchor_b, anchor_aux], axis=1)
                pos_att_aux = cat_embs * cat_contri + brand_embs * brand_contri 
                pos_att_aux *= self.auxiliary_mask
                pos_all = tf.concat([pos_cat_b, pos_brand_b, pos_att_aux], axis=1) 
            
            truncated_normal_init_cl = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=tf.sqrt(tf.div(2.0, 3 * self.num_units + 1)))
            self.att_mlp_layer = Dense(units=self.num_units, activation=None, kernel_initializer=truncated_normal_init_cl, use_bias=False)
            anchor_all_mlp = self.att_mlp_layer(anchor_all)
            pos_all_mlp = self.att_mlp_layer(pos_all)
            
            if self.dataset == 'Tmall':
                cat_p_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_cat_b, -1), 0))  # mask purchase [bs, L]
                seller_p_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_seller_b, -1), 0))  # mask purchase [bs, L]
                brand_p_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_brand_b, -1), 0))  # mask purchase [bs, L]
                aux_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_att_aux, -1), 0)) # mask favorite and view [bs, L]
                neg_mask = tf.concat([cat_p_mask, seller_p_mask, brand_p_mask, aux_mask], axis=-1)
            else:
                cat_p_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_cat_b, -1), 0)) 
                brand_p_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_brand_b, -1), 0))
                aux_mask = tf.to_float(tf.not_equal(tf.reduce_sum(pos_att_aux, -1), 0))       
                neg_mask = tf.concat([cat_p_mask, brand_p_mask, aux_mask], axis=-1) 
            self.neg_logits = self.neg_sample_gather_for_cl(anchor_all_mlp, pos_all_mlp, neg_mask, args.random_seed) 
            pos_gat = self.gather_indexes(pos_all_mlp, positions_mask=neg_mask)
            anchor_gat = self.gather_indexes(anchor_all_mlp, positions_mask=neg_mask)
            self.pos_logits = tf.reshape(tf.reduce_sum(tf.multiply(anchor_gat, pos_gat), axis=-1), [-1, 1])
            
            all_logits = tf.concat([tf.expand_dims(self.pos_logits, 1), tf.expand_dims(self.neg_logits, 1)], axis=-1)
            all_logits = tf.sigmoid(all_logits)
            log_prob = tf.nn.log_softmax(all_logits, -1)[:, :, 0]
            self.loss_contrastive_att = -tf.reduce_mean(tf.reduce_sum(log_prob, 1), 0)
        else:
            self.loss_contrastive_att = tf.constant(0.0, dtype=tf.float32)
       
        if 'c' in self.ext_modules:
            truncated_normal_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=tf.sqrt(tf.div(2.0, 3 * self.num_units + 1)))
            self.bsm_mlp_layer = Dense(units=self.num_units, activation=None, kernel_initializer=truncated_normal_init, use_bias=False)
            anchor_bsm_w = self.bsm_mlp_layer(mmoe_outputs)
            posneg_bsm_w = self.bsm_mlp_layer(item_embs)
            p_mask_bsm = tf.to_float(tf.equal(self.behavior_id, 0))
            anchor_all_bsm = self.gather_indexes(anchor_bsm_w, positions_mask=p_mask_bsm)
            pos_all_bsm = self.gather_indexes(posneg_bsm_w, positions_mask=p_mask_bsm)
            self.positive_bsm = tf.multiply(anchor_all_bsm, pos_all_bsm) 
            self.pos_logits_bsm = tf.reshape(tf.reduce_sum(self.positive_bsm, -1), [-1, 1]) 
            anc_bsm = anchor_bsm_w * tf.expand_dims(p_mask_bsm, -1)
            pos_bsm = posneg_bsm_w * tf.expand_dims(p_mask_bsm, -1)
            self.neg_logits_bsm = self.neg_sample_gather_for_cl(anc_bsm, pos_bsm, p_mask_bsm, args.random_seed)
            all_logits_bsm = tf.concat([tf.expand_dims(self.pos_logits_bsm, 1), tf.expand_dims(self.neg_logits_bsm, 1)], 2)
            all_logits_bsm = tf.sigmoid(all_logits_bsm)
            
            log_prob_bsm = tf.nn.log_softmax(all_logits_bsm, -1)[:, :, 0]
            self.loss_contrastive_bsmmoe = -tf.reduce_mean(tf.reduce_sum(log_prob_bsm, 1), 0)
        else:
            self.loss_contrastive_bsmmoe = tf.constant(0.0, dtype=tf.float32)
            
        outputs_pos = outputs
        outputs_neg = outputs

        # ==== training loss ====
        positive_rating = tf.reduce_sum(outputs_pos * positive_embs, -1)
        negative_rating = tf.reduce_sum(outputs_neg * negative_embs, -1)
        flag_exist = tf.to_float(tf.not_equal(self.pos_id, 0))
        self.loss = tf.reduce_sum(- tf.math.log(tf.sigmoid(positive_rating) + 1e-24) * flag_exist
                                  - tf.math.log(1 - tf.sigmoid(negative_rating) + 1e-24) * flag_exist) / tf.reduce_sum(flag_exist)
        reg_loss = tf.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_loss)
        self.loss_item = self.loss
        att_loss_rate = args.att_cl_rate
        bsmmoe_loss_rate = args.bsmmoe_cl_rate
        self.att_loss_cl = self.loss_contrastive_att * att_loss_rate
        self.bsmmoe_loss_cl = self.loss_contrastive_bsmmoe * bsmmoe_loss_rate
        self.loss += self.att_loss_cl
        self.loss += self.bsmmoe_loss_cl

        # ==== test prediction ====
        self.candidate_id = tf.placeholder(tf.int32, shape=(None, self.num_item))
        candidate_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.candidate_id)
        self.cand_rating = tf.reduce_sum(outputs_te * candidate_embs, -1)
        # self.cand_rating = tf.nn.softmax(self.cand_rating)
        # ==== optimizer ====
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


    # ======== multihead attention ========
    def multihead_attention(self, inputs, padding_mask=None,
                            num_heads=2, dropout_rate=None, score='scaled_dot',
                            causality=True, get_att='last_ave', residual=True):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        with tf.compat.v1.variable_scope("multihead_attention"):
            # linear projections, shape: [batch_size, seq_length, num_units]
            if score == 'location':
                weights = tf.compat.v1.get_variable('weights',
                                                    dtype=tf.float32,
                                                    shape=[1, self.num_units],
                                                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
                if causality == False:  # obtain output shape: [batch_size, 1, num_units]
                    Q = tf.tile(tf.expand_dims(weights, 0), [tf.shape(inputs)[0], 1, 1])
                else:  # obtain output shape: [batch_size, seq_length, num_units]
                    Q = tf.tile(tf.expand_dims(weights, 0), [tf.shape(inputs)[0], tf.shape(inputs)[1], 1])
            else:
                Q = tf.layers.dense(inputs, self.num_units, activation=None)

            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)

            # split and place in parallel, shape: [batch_size * num_heads, seq_length, num_units / num_heads]
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # score function, shape: [batch_size * num_heads, seq_length_Q, seq_length_K]
            if score == 'scaled_dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)
            elif score == 'location' or score == 'dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1]))

            # causality masking
            if causality == True:
                tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(outputs[0, :, :])).to_dense()
                # #v for tf.__version__='1.2.1'
                # tril = tf.contrib.linalg.LinearOperatorTriL(tf.ones_like(outputs[0, :, :])).to_dense()
                causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                outputs = tf.where(tf.equal(causality_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)

            # key masking
            if padding_mask is not None:
                key_mask = tf.transpose(tf.tile(padding_mask, [num_heads, 1, tf.shape(Q)[1]]), [0, 2, 1])
                outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)

            # softmax normalization
            outputs = tf.nn.softmax(outputs)

            # query masking
            if (score != 'location') and (padding_mask is not None):
                query_mask = tf.tile(padding_mask, [num_heads, 1, tf.shape(inputs)[1]])
                outputs = outputs * query_mask

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            # attention_a = outputs

            # return attention for visualization
            if get_att == 'lastq_ave':
                attention = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            if get_att == 'lastq_multi':
                attention = tf.split(outputs[:, -1], num_heads, axis=0)
            elif get_att == 'batch_multi':
                attention = outputs

            # weighted sum, shape: [batch_size * num_heads, seq_length_Q, num_units / num_heads]
            outputs = tf.matmul(outputs, Vh)

            # concatenate different heads, shape: [batch_size, seq_length_Q, num_units]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        if residual == True:
            outputs += inputs

        return outputs, attention

    # ======== multihead attention ========
    def multihead_attention_ub(self, inputs, loc_embs, attention_mask=None,
                               num_heads=2, dropout_rate=None, score='scaled_dot',
                               causality=True, get_att='last_ave', residual=True):
        with tf.variable_scope("multihead_attention"):
            # linear projections, shape: [batch_size, seq_length+4, num_units]
            Q = tf.layers.dense(inputs, self.num_units, activation=None)
            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)

            # split and place in parallel, shape: [batch_size * num_heads, seq_length, num_units / num_heads]
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # score function, shape: [batch_size * num_heads, seq_length_Q, seq_length_K]
            if score == 'scaled_dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)

            outputs = tf.where(tf.equal(attention_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)

            outputs = tf.nn.softmax(outputs)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # return attention for visualization
            if get_att == 'lastq_ave':
                attention = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            if get_att == 'lastq_multi':
                attention = tf.split(outputs[:, -1], num_heads, axis=0)
            elif get_att == 'batch_multi':
                attention = outputs

            # weighted sum, shape: [batch_size * num_heads, seq_length_Q, num_units / num_heads]
            outputs = tf.matmul(outputs, Vh)

            # concatenate different heads, shape: [batch_size, seq_length_Q, num_units]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        if residual == True:
            outputs += inputs

        return outputs, attention
    # ======== feed forward ========

    def feed_forward(self, inputs,
                     inner_units=None, dropout_rate=None):
        if inner_units is None:
            inner_units = self.num_units
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        with tf.compat.v1.variable_scope("feed_forward"):
            # inner layer
            params = {"inputs": inputs, "filters": inner_units, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # readout layer
            params = {"inputs": outputs, "filters": self.num_units, "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        outputs += inputs

        return outputs

    # ======== layer normalization ========
    def layer_normalization(self, inputs, epsilon=1e-8):
        with tf.variable_scope("layer_normalization"):
            alpha = tf.Variable(tf.ones(self.num_units))
            beta = tf.Variable(tf.zeros(self.num_units))

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = alpha * normalized + beta

        return outputs
    
    def layer_normalization_anyhidden(self, inputs, num_units=None, epsilon=1e-8):
        with tf.compat.v1.variable_scope("layer_normalization_anyhidden"):
            alpha = tf.Variable(tf.ones(num_units))
            beta = tf.Variable(tf.zeros(num_units))

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = alpha * normalized + beta

        return outputs
    
    def create_behavior_attention_mask(self, padding_mask=None, causality=True, behavior='v'):
        """
        Create 3D attention mask from a 2D tensor mask.
        v: view
        b: buy
        f: fav
        """
        batch_size = tf.shape(padding_mask)[0]
        seq_length = tf.shape(padding_mask)[1]
        to_mask = tf.cast(
            tf.reshape(padding_mask, [batch_size, 1, seq_length]), tf.float32)

        broadcast_ones = tf.ones(
            shape=[batch_size, seq_length, 1], dtype=tf.float32)

        mask = broadcast_ones * to_mask  # [bs,L,L]
        # causality masking
        if causality == True:
            tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(mask[0, :, :])).to_dense()
            mask_causality = tf.tile(tf.expand_dims(tril, 0), [batch_size, 1, 1])
            mask = mask * mask_causality

        behavior_mask = None
        if behavior == 'v':
            behavior_mask = tf.cast(
                tf.reshape(self.view_mask, [batch_size, 1, self.maxlen]), tf.float32)
        if behavior == 'b':
            behavior_mask = tf.cast(
                tf.reshape(self.purchase_mask, [batch_size, 1, self.maxlen]), tf.float32)
        if behavior == 'f':
            behavior_mask = tf.cast(
                tf.reshape(self.favorite_mask, [batch_size, 1, self.maxlen]), tf.float32)
        mask = mask * behavior_mask
        return mask

    def get_ubp_memory(self, self_att_output):
        """Gather dynamic user_behavior preference or context information in a form of memory matrix"""
        batch_size = tf.shape(self_att_output)[0]
        padding = tf.expand_dims(tf.zeros([1, self.num_units]), 1)
        padding = tf.tile(padding, [batch_size, 1, 1])
        tensor = tf.concat([padding, self_att_output], -2)  # [bs,L+1,d]

        mask0 = tf.to_float(tf.equal(self.behavior_id, 0))  # mask purchase [bs, L]
        mask2 = tf.to_float(tf.equal(self.behavior_id, 2))  # mask favorite
        mask3 = tf.to_float(tf.equal(self.behavior_id, 3))  # mask view

        def get_one_memory(mask):
            pad_mask = tf.ones([batch_size, 1])
            mask_with_pad = tf.concat([pad_mask, mask], -1)
            idx_bias = tf.reshape(tf.reduce_sum(mask, -1), [-1]) + 1  # [bs,1]
            idx_bias = tf.reshape(tf.cast(tf.cumsum(idx_bias, axis=-1, exclusive=True), tf.int32),
                                  [batch_size, 1])  # [bs,1]
            mask = tf.cast(tf.cumsum(mask, axis=-1), tf.int32)  # [bs, L]
            mask = mask + idx_bias  # [bs, L]
            position = tf.where(tf.equal(mask_with_pad, 1))
            embedding = tf.gather_nd(tensor, position)
            return tf.gather(embedding, mask)

        outputs0 = get_one_memory(mask0)
        outputs2 = get_one_memory(mask2)
        outputs3 = get_one_memory(mask3)
        return outputs0, outputs2, outputs3
    
    
    def get_target_view(self, b, f, v):
        target_purchase_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 0)), -1)  # mask purchase
        target_favorite_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 2)), -1)  # mask favorite
        target_view_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 3)), -1)  # mask view

        b *= target_purchase_mask
        f *= target_favorite_mask
        v *= target_view_mask
        
        output = b + f + v
        return output

    def gather_indexes(self, sequence_tensor, positions_mask):
        """Gathers the vectors at the specific positions over a minibatch."""
        position = tf.where(tf.equal(positions_mask, 1))
        output_tensor = tf.gather_nd(sequence_tensor, position)
        return output_tensor
    
    def repeat(self, input, mask):
        """Gather tensor in a repeated way."""
        repeats = tf.to_float(tf.equal(mask, 1))
        repeats = tf.reduce_sum(repeats, axis=1)
        repeats = tf.cast(repeats, tf.int32)
        max_repeat = gen_math_ops.maximum(
            0, gen_math_ops._max(repeats, _all_dimensions(repeats)))
        mask = sequence_mask(repeats, max_repeat)
        expanded = tf.expand_dims(input, 1)
        multiples = [1] * expanded.shape.ndims
        multiples[1] = max_repeat
        tiled = tf.tile(expanded, multiples)
        return boolean_mask(tiled, mask)
    
    def multihead_attention_cat(self, query, inputs, padding_mask=None,
                            num_heads=1, dropout_rate=None, score='scaled_dot',
                            causality=True, get_att='last_ave', residual=True):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
        with tf.compat.v1.variable_scope("multihead_attention_cat"):
            Q = tf.layers.dense(inputs, self.num_units, activation=None)
            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
            outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)
            tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(outputs[0, :, :])).to_dense()
            causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            outputs = tf.where(tf.equal(causality_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)
            key_mask = tf.transpose(tf.tile(padding_mask, [num_heads, 1, tf.shape(Q)[1]]), [0, 2, 1])
            outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)
            outputs = tf.nn.softmax(outputs)
            query_mask = tf.tile(padding_mask, [num_heads, 1, tf.shape(inputs)[1]])
            outputs = outputs * query_mask
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            attention_lastq = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            attention_lastmulti = tf.split(outputs[:, -1], num_heads, axis=0)
            attention_batchmulti = outputs
            outputs = tf.matmul(outputs, Vh)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
        outputs += inputs
        return outputs, attention_lastq, attention_lastmulti, attention_batchmulti
    
    def multihead_attention_bran(self, query, inputs, padding_mask=None,
                            num_heads=1, dropout_rate=None, score='scaled_dot',
                            causality=True, get_att='last_ave', residual=True):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
        with tf.compat.v1.variable_scope("multihead_attention_bran"):
            Q = tf.layers.dense(inputs, self.num_units, activation=None)
            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
            outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)
            tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(outputs[0, :, :])).to_dense()
            causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            outputs = tf.where(tf.equal(causality_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)
            key_mask = tf.transpose(tf.tile(padding_mask, [num_heads, 1, tf.shape(Q)[1]]), [0, 2, 1])
            outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)
            outputs = tf.nn.softmax(outputs)
            query_mask = tf.tile(padding_mask, [num_heads, 1, tf.shape(inputs)[1]])
            outputs = outputs * query_mask
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            attention_lastq = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            attention_lastmulti = tf.split(outputs[:, -1], num_heads, axis=0)
            attention_batchmulti = outputs
            outputs = tf.matmul(outputs, Vh)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
        outputs += inputs
        return outputs, attention_lastq, attention_lastmulti, attention_batchmulti
    
    def multihead_attention_sell(self, query, inputs, padding_mask=None,
                            num_heads=1, dropout_rate=None, score='scaled_dot',
                            causality=True, get_att='last_ave', residual=True):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
        with tf.compat.v1.variable_scope("multihead_attention_sell"):
            Q = tf.layers.dense(inputs, self.num_units, activation=None)
            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
            outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)
            tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(outputs[0, :, :])).to_dense()
            causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            outputs = tf.where(tf.equal(causality_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)
            key_mask = tf.transpose(tf.tile(padding_mask, [num_heads, 1, tf.shape(Q)[1]]), [0, 2, 1])
            outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)
            outputs = tf.nn.softmax(outputs)
            query_mask = tf.tile(padding_mask, [num_heads, 1, tf.shape(inputs)[1]])
            outputs = outputs * query_mask
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            attention_lastq = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            attention_lastmulti = tf.split(outputs[:, -1], num_heads, axis=0)
            attention_batchmulti = outputs
            outputs = tf.matmul(outputs, Vh)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
        outputs += inputs
        return outputs, attention_lastq, attention_lastmulti, attention_batchmulti
    
    def dynamic_att_attention(self, query, inputs, dropout_rate=None):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
        with tf.variable_scope("att_attention", reuse=tf.AUTO_REUSE):
            query = tf.reshape(query, [-1, self.maxlen, 1, self.num_units])
            Q = tf.layers.dense(query, self.num_units, activation=None)
            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)
            outputs = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / (tf.to_float(tf.shape(K))[-1] ** 0.5)
            outputs = tf.nn.softmax(outputs)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            outputs = tf.matmul(outputs, V)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            outputs = tf.reshape(outputs, [-1, self.maxlen, self.num_units])
        return outputs
    
    def neg_sample_gather_for_cl(self, anchor_all, pos_all, neg_mask, random_seed):
        batch_size = tf.shape(pos_all)[0]
        def fill_pos(i):
            def no_fill():
                return pos_all[i]
            def empty_fill():
                non_mask_values = tf.boolean_mask(pos_all[i], tf.not_equal(neg_mask[i], 0))
                random_indices = tf.random_uniform(shape=[tf.shape(pos_all[i])[0]], minval=0,
                                                    maxval=tf.shape(non_mask_values)[0], dtype=tf.int32)
                random_fill_values = tf.gather(non_mask_values, random_indices)
                return tf.where(tf.equal(neg_mask[i], 0), random_fill_values, pos_all[i])
            filled_tensor = tf.cond(tf.reduce_sum(neg_mask[i]) > 0.0, empty_fill, no_fill)
            return filled_tensor
        
        pos_all_fill = tf.map_fn(fill_pos, elems=tf.range(batch_size), dtype=tf.float32)
        neg_all = tf.map_fn(lambda x: tf.roll(pos_all_fill, shift=x + 1, axis=0), elems=tf.range(batch_size-1), dtype=tf.float32)
        neg_all = tf.transpose(neg_all, perm=[1, 2, 0, 3])
        neg_shape = tf.shape(neg_all)[2]
        anchor_input_all = tf.tile(anchor_all[:, :, tf.newaxis, :], [1, 1, neg_shape, 1])
        neg_sel = self.gather_indexes(neg_all, positions_mask=neg_mask)
        anchor_sel = self.gather_indexes(anchor_input_all, positions_mask=neg_mask)
        neg_output = tf.multiply(anchor_sel, neg_sel)
        neg_outputs = tf.reshape(tf.reduce_sum(neg_output, -1),
                                 [-1, batch_size - 1])
        return neg_outputs

import tensorflow as tf
import numpy as np
import time

from PTBReader import *


class LSTMLanguageModel(object):
    def __init__(self, sess, hparams, mode):
        self.sess = sess
        self.hparams = hparams

        print(self.hparams)
        self.mode = mode

        self.vocab_size = self.hparams.vocab_size

        self.reader = PTBReader(data_dir="../data",dataset_name="ptb", vocab_size=10000)

    def define_graph(self):
        filenames = ["../data/ptb_train_0.tfrecords"]
        #batch_size = 10
        min_after_dequeue = 1000

        filename_queue = tf.train.string_input_producer(
            filenames)
        input_sequence, output_sequence, input_shape, output_shape = self.reader.read_tf_record_file(filename_queue)

        input_seq_batch, output_seq_batch, input_shape_batch, output_shape_batch = tf.train.shuffle_batch(
            [input_sequence, output_sequence, input_shape, output_shape], batch_size=self.hparams.batch_size,
            capacity=min_after_dequeue * 3 + 1, min_after_dequeue=min_after_dequeue)

        dense_input_seq_batch = tf.sparse_to_dense(sparse_indices=input_seq_batch.indices,
                                                   output_shape=input_seq_batch.dense_shape,
                                                   sparse_values=input_seq_batch.values,
                                                   default_value=0,
                                                   validate_indices=True,
                                                   name=None)
        dens_output_seq_batch = tf.sparse_to_dense(sparse_indices=output_seq_batch.indices,
                                                   output_shape=output_seq_batch.dense_shape,
                                                   sparse_values=output_seq_batch.values,
                                                   default_value=0,
                                                   validate_indices=True,
                                                   name=None)

        input_seq_lengths = tf.reshape(input_shape_batch, [self.hparams.batch_size])
        output_seq_lengths = tf.reshape(output_shape_batch, [self.hparams.batch_size])

        max_output_length = tf.reduce_max(output_seq_lengths)

        self.y = tf.stack([dens_output_seq_batch[i] for i in range(self.hparams.batch_size)])


        self.embedding = tf.get_variable("embedding", [self.vocab_size, self.hparams.number_of_hidden_units],
                                         dtype=tf.float32)
        embedded_inputs = [tf.nn.embedding_lookup(self.embedding, dense_input_seq_batch[i]) for i in range(self.hparams.batch_size)]
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            embedded_inputs = tf.nn.dropout(embedded_inputs, self.hparams.keep_prob)

        self.cell, _ = self._get_lstm_cell()
        tf.logging.info(dense_input_seq_batch)
        tf.logging.info(input_seq_lengths)
        all_states, current_state = tf.nn.dynamic_rnn(inputs=embedded_inputs,
                                                      cell=self.cell,
                                                      sequence_length=input_seq_lengths,
                                                      initial_state=None,
                                                      time_major=False,
                                                      scope=None,
                                                      dtype=tf.float32)

        tf.logging.info(all_states)
        self.sequence_predictions = all_states
        tf.logging.info(self.sequence_predictions)
        for sp in tf.unstack(self.sequence_predictions):
            tf.logging.info(sp)

        self.projection = tf.get_variable("output_projection", [self.hparams.number_of_hidden_units, self.vocab_size],dtype=tf.float32, initializer=tf.zeros_initializer())
        self.projection_bias = tf.get_variable("output_projection_bias", [self.vocab_size],dtype=tf.float32, initializer=tf.zeros_initializer())
        self.projected_seq_predictions = [tf.matmul(sp,self.projection) for sp in tf.unstack(self.sequence_predictions)]




        #cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.projected_seq_predictions, labels=self.y)
        #tf.logging.info(cross_ent)
        
        tf.logging.info(self.sequence_predictions[:,-1])
        
        predictions = tf.concat(tf.unstack(self.sequence_predictions),axis=0)
        tf.logging.info(predictions)
        y = tf.concat(tf.unstack(self.y),axis=0)
        nce_loss = tf.nn.nce_loss(weights=tf.transpose(self.projection),
                                  biases=self.projection_bias,
                                  inputs= predictions,#self.sequence_predictions[0], 
                                  labels=tf.expand_dims(y,1),#self.y[0],1),
                                  num_sampled=20, 
                                  num_classes=10000, 
                                  num_true=1, 
                                  sampled_values=None, 
                                  remove_accidental_hits=False, partition_strategy='div', name='nce_loss')# for y,sp in zip(tf.unstack(self.y),tf.unstack(self.sequence_predictions))]
        
        
        self.train_loss = (tf.reduce_sum(nce_loss) /
                           self.hparams.batch_size)
        tf.logging.info(self.train_loss)
        tf.summary.scalar("loss", tf.reduce_mean(self.train_loss))

        correct_prediction = tf.equal(self.y, tf.argmax(self.projected_seq_predictions, 2))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", self.accuracy)



    def _get_lstm_cell(self):
        cell = tf.contrib.rnn.LSTMBlockCell(self.hparams.number_of_hidden_units, forget_bias=0.0)

        # fw_cell = tf.contrib.rnn.ResidualWrapper(fw_cell)
        # bw_cell = tf.contrib.rnn.ResidualWrapper(bw_cell)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.hparams.keep_prob)

        # elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
        # else: #if self.mode == tf.contrib.learn.ModeKeys.INFER:
        stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.hparams.depth, state_is_tuple=True)

        initial_state = stacked_cell.zero_state(self.hparams.batch_size, tf.float32)

        return stacked_cell, initial_state


    def _define_train(self):
        """warmup_steps = self.hparams.learning_rate_warmup_steps
        warmup_factor = self.hparams.learning_rate_warmup_factor
        """
        """print("  start_decay_step=%d, learning_rate=%g, decay_steps %d, "
              "decay_factor %g, learning_rate_warmup_steps=%d, "
              "learning_rate_warmup_factor=%g, starting_learning_rate=%g" %
              (self.hparams.start_decay_step, self.hparams.learning_rate, self.hparams.decay_steps,
               self.hparams.decay_factor, warmup_steps, warmup_factor,
               (self.hparams.learning_rate * warmup_factor ** warmup_steps)))
        """
        self.global_step = tf.Variable(0, trainable=False)



        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            starter_learning_rate = self.hparams.learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       1000, 0.96, staircase=True)
            #inv_decay = warmup_factor ** (
            #    tf.to_float(warmup_steps - self.global_step))
            #self.learning_rate = tf.cond(
            #    self.global_step < self.hparams.learning_rate_warmup_steps,
            #    lambda: inv_decay * self.learning_rate,
            #    lambda: self.learning_rate,
            #    name="learning_rate_decay_warump_cond")

            if self.hparams.optimizer == "sgd":
                self.learning_rate = tf.cond(
                    self.global_step < self.hparams.start_decay_step,
                    lambda: self.learning_rate,
                    lambda: tf.train.exponential_decay(
                        self.learning_rate,
                        (self.global_step - self.hparams.start_decay_step),
                        self.hparams.decay_steps,
                        self.hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif self.hparams.optimizer == "adam":
                assert float(
                    self.hparams.learning_rate
                ) <= 0.001, "! High Adam learning rate %g" % self.hparams.learning_rate
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)




            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                params = tf.trainable_variables()
                gradients = tf.gradients(
                    self.train_loss,
                    params,
                    colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.max_gradient_norm)
                #self.update = self.optimizer.minimize(self.train_loss,global_step=self.global_step)
                self.update = self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


    def train(self, init=True):
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("../tmp/bidi", self.sess.graph)

        start_time = time.time()
        if init:
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            self.sess.run(init_g)
            self.sess.run(init_l)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(self.hparams.number_of_epochs):
            iteration = 0
            while iteration * self.hparams.batch_size < self.hparams.training_size:
                _, train_cost, train_accuracy = self.sess.run(
                    [self.update, self.train_loss, self.accuracy])

                iteration += 1
                if iteration % 10 == 0:
                    print("iterations: [%2d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                          % (iteration, time.time() - start_time, np.mean(train_cost), train_accuracy))

                    # if iteration % 1000 == 0:
                    #    self.save(global_step=self.global_step)

        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    hparams = tf.flags
    hparams.DEFINE_integer("training_size", 100000, "total number of training samples")
    hparams.DEFINE_integer("number_of_epochs", 100, "Epoch to train [25]")
    hparams.DEFINE_integer("vocab_size", 10000, "The size of vocabulary [10000]")
    hparams.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
    hparams.DEFINE_integer("depth", 2, "Depth [1]")
    hparams.DEFINE_integer("max_nsteps", 1000, "Max number of steps [1000]")
    hparams.DEFINE_integer("number_of_hidden_units", 512, "The size of hidden layers")
    hparams.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
    hparams.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
    hparams.DEFINE_float("keep_prob", 0.7, "keep_prob [0.5]")
    hparams.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
    hparams.DEFINE_string("dtype", "float32", "dtype [float32]")
    hparams.DEFINE_string("model", "LSTM", "The type of model to train and test [LSTM, BiLSTM, Attentive, Impatient]")
    hparams.DEFINE_string("data_dir", "../data", "The name of data directory [data]")
    hparams.DEFINE_string("dataset_name", "ptb", "The name of dataset [cnn, dailymail]")
    hparams.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    hparams.DEFINE_integer("learning_rate_warmup_steps", 0, "How many steps we inverse-decay learning.")
    hparams.DEFINE_float("learning_rate_warmup_factor", 1.0, "The inverse decay factor for each warmup step.")
    hparams.DEFINE_integer("start_decay_step", 10, "When we start to decay")
    hparams.DEFINE_integer("decay_steps", 10000, "How frequent we decay")
    hparams.DEFINE_float("decay_factor", 0.98, "How much we decay.")
    hparams.DEFINE_string("optimizer", "adam", "sgd | adam")
    hparams.DEFINE_bool("colocate_gradients_with_ops", True,
                        "Whether try colocating gradients with "
                        "corresponding op")
    hparams.DEFINE_float("--max_gradient_norm", 5.0, "Clip gradients to this norm.")
    hparams = hparams.FLAGS

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session() as sess:
        lstm_lm = LSTMLanguageModel(sess=sess, hparams=hparams, mode=tf.contrib.learn.ModeKeys.TRAIN)

        lstm_lm.define_graph()
        lstm_lm._define_train()
        lstm_lm.train()




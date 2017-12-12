from LSTMLanguageModel import *

if __name__ == '__main__':
    hparams = tf.flags
    hparams.DEFINE_string('exp_name', 'main_experiment', 'Name for experiment. Logs will '
                                                              'be saved in a directory with this'
                                                              ' name, under log_root.')
    hparams.DEFINE_string('log_root', '../log_root', 'Root directory for all logging.')
    hparams.DEFINE_integer("training_size", 100000, "total number of training samples")
    hparams.DEFINE_integer("number_of_epochs", 100, "Epoch to train [25]")
    hparams.DEFINE_integer("vocab_size", 10000, "The size of vocabulary [10000]")
    hparams.DEFINE_integer("batch_size", 1, "The size of batch images [32]")
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

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    hparams.log_root = os.path.join(hparams.log_root, hparams.model, hparams.exp_name)
    if not os.path.exists(hparams.log_root):
        os.makedirs(hparams.log_root)

    train_dir = os.path.join(hparams.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)


    tf.logging.set_verbosity(tf.logging.INFO)

    lstm_lm = LSTMLanguageModel(hparams=hparams,
                                mode=tf.contrib.learn.ModeKeys.TRAIN)
    lstm_lm.define_graph()
    lstm_lm._define_train()

    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                             save_model_secs=120,  # checkpoint every 60 secs
                             global_step=None
                             )
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session()
    tf.logging.info("Created session.")

    with sess_context_manager as sess:
        try:
            lstm_lm.train(sess, summary_writer)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
            sv.stop()






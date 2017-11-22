import tensorflow as tf
import numpy as np
from tqdm import *

from glob import glob
from gensim import corpora
from nltk.tokenize import RegexpTokenizer


import time
import os
import re

class PTBReader(object):
    _WORD_SPLIT = re.compile("([.,!?\"':;)(])")
    _DIGIT_RE = re.compile(r"(^| )\d+")
    tokenizer = RegexpTokenizer(r'@?\w+')

    _BAR = "_BAR"
    _UNK = "_UNK"
    _EOS = "<EOS>"

    _START_VOCAB = [_BAR, _UNK]

    def __init__(self,data_dir,dataset_name, vocab_size):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size
        self.EOS_ID = self.vocab_size - 1
        self.UNK_ID = self.vocab_size - 2

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    def basic_tokenizer(self,sentence):
        """Very basic tokenizer: split the sentence into a list of tokens."""
        words = PTBReader.tokenizer.tokenize(sentence)
        return [w for w in words] # if w not in PTBReader.cachedStopWords]

    def read_words(self):
        worded_sentences = []
        with tf.gfile.GFile(self.filename,"r") as f:
            sentences = f.read().split("\n")
            worded_sentences = [ sent.split() for sent in sentences]
            for i in np.arange(len(worded_sentences)):
                worded_sentences[i].append("<EOS>")

        print(worded_sentences[0])

    def initialize_vocabulary(self,vocabulary_path):
        """Initialize vocabulary from file.
        We assume the vocabulary is stored one-item-per-line, so a file:
          dog
          cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].
        Args:
          vocabulary_path: path to the file containing the vocabulary.
        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).
        Raises:
          ValueError: if the provided vocabulary_path does not exist.
        """
        if tf.gfile.Exists(vocabulary_path):
            vocab = corpora.Dictionary.load(vocabulary_path)
            print("vocab length: ",len(vocab.token2id))

            return vocab.token2id, vocab.token2id.keys()
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)

    def build_vocab_file(self):
        train_path = os.path.join(self.data_dir, self.dataset_name,"train")

        context_fname = os.path.join(self.data_dir, self.dataset_name, '%s.context' % self.dataset_name)
        vocab_fname = os.path.join(self.data_dir, self.dataset_name, '%s.vocab%s' % (self.dataset_name, self.vocab_size))

        if not os.path.exists(context_fname):
            print(" [*] Combining all contexts for %s in %s ..." % (self.dataset_name, train_path))
            context = self.get_all_context(train_path, context_fname)
        else:
            context = tf.gfile.GFile(context_fname, mode="r").read()
            print(" [*] Skip combining all contexts")

        if not os.path.exists(vocab_fname):
            t0 = time.time()
            print("Creating vocabulary %s" % (vocab_fname))
            print("max_vocabulary_size: ", self.vocab_size)
            texts = [word for word in context.lower().split()]# if word not in DataReader.cachedStopWords]
            dictionary = corpora.Dictionary([texts], prune_at=self.vocab_size-2)
            dictionary.filter_extremes(no_below=1, no_above=1, keep_n=self.vocab_size-2)

            print("vocab length: ", len(dictionary.token2id))
            print("Tokenize : %.4fs" % (t0 - time.time()))
            dictionary.save(vocab_fname)

        print(" [*] Convert data in %s into vocab indicies..." % (train_path))
        self.questions_to_token_ids(train_path, vocab_fname, self.vocab_size)


    def questions_to_token_ids(self,data_path, vocab_fname, vocab_size):
        vocab, _ = self.initialize_vocabulary(vocab_fname)
        for fname in tqdm(glob(os.path.join(data_path, "*.txt"))):
            self.data_to_token_ids(fname, fname + ".ids%s" % vocab_size, vocab)


    def data_to_token_ids(self,data_path, target_path, vocab,
                          tokenizer=None, normalize_digits=True):
        """Tokenize data file and turn into token-ids using given vocabulary file.
        This function loads data line-by-line from data_path, calls the above
        sentence_to_token_ids, and saves the result to target_path. See comment
        for sentence_to_token_ids on the details of token-ids format.
        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        # if not gfile.Exists(target_path):
        if True:
            with tf.gfile.GFile(data_path, mode="r") as data_file:
                counter = 0
                results = []
                for line in data_file:
                    token_ids = self.sentence_to_token_ids(line, vocab, tokenizer,
                                                          normalize_digits)
                    results.append(" ".join([str(tok) for tok in token_ids]) + "\n")
                try:
                    len_d, len_q = len(results[2].split()), len(results[4].split())
                except:
                    return
                with open("%s_%s" % (target_path, len_d + len_q), mode="w") as tokens_file:
                    tokens_file.writelines(results)

    def sentence_to_token_ids(self,sentence, vocabulary,
                              tokenizer=None, normalize_digits=True):
        """Convert a string to list of integers representing token-ids.
        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
        Args:
          sentence: a string, the sentence to convert to token-ids.
          vocabulary: a dictionary mapping tokens to integers.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        Returns:
          a list of integers, the token-ids for the sentence.
        """
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self.basic_tokenizer(sentence)

        if not normalize_digits:
            return [vocabulary.get(w, self.UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(re.sub(PTBReader._DIGIT_RE, " ", w), self.UNK_ID) for w in words]


    def get_all_context(self,dir_name, context_fname):
        context = ""
        for fname in tqdm(glob(os.path.join(dir_name, "*.txt"))):
            with open(fname) as f:
                try:
                    lines = f.read().split("\n")

                    context += (" ").join(lines)
                except:
                    print(" [!] Error occured for %s" % fname)
        print(" [*] Writing %s ..." % context_fname)
        with open(context_fname, 'w') as f:
            f.write(context)
        return context


    def convert2TFRecords(self,filenames,mode):
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        key, example = reader.read(filename_queue)
        parsed_example = tf.string_split([example], '\n')
        filename = os.path.join("../data", "cnn_" + mode + "_0" + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)

        with tf.Session() as sess:
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(len(filenames)):
                # Retrieve a single instance:
                if i > 0 and (i % 1000 == 0):
                    writer.close()
                    filename = os.path.join("../data", "ptb_train_" + "_" + str(i) + '.tfrecords')
                    writer = tf.python_io.TFRecordWriter(filename)

                (_, data, _) = sess.run(parsed_example)
                print(data[0])
                sentences = [list(map(int, d.decode().split(" "))) + [self.EOS_ID] for d in data]


                print(sentences[0])

                for sent in sentences:
                    feature_list = {
                        'input_seq': PTBReader._int64_feature(sent[0:len(sent) - 1 ]),
                        'output_seq': PTBReader._int64_feature(sent[1:])
                    }

                    feature = tf.train.Features(feature=feature_list)
                    example = tf.train.Example(features=feature)

                    writer.write(example.SerializeToString())

            writer.close()
            coord.request_stop()
            coord.join(threads)

    def read_tf_record_file(self,filename_queue):
        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)

        features = {
            'input_seq': tf.VarLenFeature(tf.int64),
            'output_seq': tf.VarLenFeature(tf.int64),
        }

        parsed_example = tf.parse_single_example(serialized_example, features=features)

        return parsed_example['input_seq'], parsed_example['output_seq'], \
               parsed_example['input_seq'].dense_shape, parsed_example['output_seq'].dense_shape

    def reader(self):
        filenames = ["../data/ptb_train_0.tfrecords"]
        batch_size = 10
        min_after_dequeue = 1000

        filename_queue = tf.train.string_input_producer(
            filenames)
        input_sequence, output_sequence, input_shape, output_shape = self.read_tf_record_file(filename_queue)

        input_seq_batch, output_seq_batch, input_shape_batch, output_shape_batch = tf.train.shuffle_batch(
            [input_sequence, output_sequence, input_shape, output_shape], batch_size=batch_size,
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

        input_seq_lengths = tf.reshape(input_shape_batch, [batch_size])
        output_seq_lengths = tf.reshape(output_shape_batch, [batch_size])

        with tf.Session() as sess:
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(100):
                print(i)
                [input, output, input_lengths,output_length] = sess.run([dense_input_seq_batch,dens_output_seq_batch,input_seq_lengths,output_seq_lengths])

                #print(input[0])
                print(input_lengths[0]," ",output_length[0])
                print(input[0])
                print(output[0])

            coord.request_stop()
            coord.join(threads)




if __name__ == '__main__':
    ptb_reader = PTBReader(data_dir="../data",dataset_name="ptb", vocab_size=10000)
    #ptb_reader.build_vocab_file() #read_words()

    mode = "train"
    train_files = glob(os.path.join("../data", "ptb",
                                    mode, "*.txt.ids%s_*" % (10000)))
    #ptb_reader.convert2TFRecords(filenames=train_files,mode=mode)

    ptb_reader.reader()
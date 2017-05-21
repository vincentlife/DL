import tensorflow as tf
import numpy as np
import sys, time
from copy import deepcopy
from tensorflow.contrib.seq2seq import sequence_loss
from collections import defaultdict
from tutorial.datareader import ptb_iterator, get_ptb_dataset


class Config():
    batch_size = 64  # 每批样本数量
    embed_size = 50  # 词向量维数
    hidden_size = 100 # RNN状态维数
    num_steps = 10 # 每步batch数
    max_epochs = 16  # 最大epoch次数
    early_stopping = 2
    dropout = 0.9
    lr = 0.001


class Vocab():
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)
        self.vocab_size = 1

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        self.vocab_size = len(self.word_freq)
        print('{} total words with {} uniques'.format(self.total_words, self.vocab_size))

    def encode(self, word):
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class RNNLM():
    def load_data(self, debug=False):
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset("train"))
        self.encoded_train = np.array([self.vocab.encode(word) for word in get_ptb_dataset("train")], dtype=np.int32)
        self.encoded_valid = np.array([self.vocab.encode(word) for word in get_ptb_dataset("valid")], dtype=np.int32)
        self.encoded_test = np.array([self.vocab.encode(word) for word in get_ptb_dataset("test")], dtype=np.int32)
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projection(self.rnn_outputs)
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        output = tf.concat([tf.expand_dims(output,1) for output in self.outputs],1)
        self.calculate_loss = self.add_loss_up(output=output)
        # self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.num_steps])
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.num_steps])
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def add_embedding(self):
        '''
        input (batch_size,num_steps)
        :return: inputs: ([(batch_size,embed_size),(batch_size,embed_size)])
        '''
        with tf.device('/cpu:0'):
            embedding_dict = tf.get_variable("Embedding", [self.vocab.vocab_size, self.config.embed_size],
                                             initializer=tf.random_uniform_initializer)
            # L: (vocab_size, embed_size) input (batch_size,num_steps) result (batch_size,num_steps,embed_size)
            inputs = tf.nn.embedding_lookup(embedding_dict, self.input_placeholder)
            # inputs: ([(batch_size,embed_size),(batch_size,embed_size)])
            inputs = [tf.squeeze(x, [1]) for x in tf.split(inputs, self.config.num_steps, 1)]
            return inputs

    def add_model(self, inputs):
        '''
        :param inputs: [(batch_size,embed_size),(batch_size,embed_size)]  length: num_step
        :return: [(batch_size,hidden_size),(batch_size,hidden_size),]
        '''
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]
        with tf.variable_scope("RNN") as scope:
            self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            rnn_outputs = []
            for step, current_input in enumerate(inputs):
                if step > 0:
                    scope.reuse_variables()
                # history weights matrix
                RNN_H = tf.get_variable("HMatrix", [self.config.hidden_size, self.config.hidden_size])
                # input weights matrix
                RNN_I = tf.get_variable("IMatrix", [self.config.embed_size, self.config.hidden_size])
                # bias
                RNN_b = tf.get_variable("bias", [self.config.hidden_size])
                state = tf.nn.sigmoid(
                    tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I) + RNN_b)
                rnn_outputs.append(state)
            self.final_state = state

        rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]
        return rnn_outputs

    def add_projection(self, rnn_outputs):
        '''
        :param rnn_outputs: [(batch_size,hidden_size),(batch_size,hidden_size),]
        :return: [(batch_size,vocab_size),(batch_size,vocab_size),]
        '''
        with tf.variable_scope("Projection"):
            U = tf.get_variable("U", [self.config.hidden_size, self.vocab.vocab_size])
            b_2 = tf.get_variable("b_2", [self.vocab.vocab_size])
            outputs = [tf.matmul(o, U) + b_2 for o in rnn_outputs]
        return outputs

    def add_loss_up(self, output):
        '''
        :param output: shape (batch_size, num_step, vocab_size)
        :return: loss
        '''
        allones = tf.ones([self.config.batch_size, self.config.num_steps])

        cross_entropy = sequence_loss(output, self.labels_placeholder, allones)
        tf.add_to_collection("total_loss", cross_entropy)
        loss = tf.add_n(tf.get_collection("total_loss"))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op


    def run_epoch(self, session, data, train_op=None, verbose=50):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        state = self.initial_state.eval()
        total_loss = []
        for step, (x, y) in enumerate(ptb_iterator(data, config.batch_size, config.num_steps)):
            feed_dict = {self.input_placeholder: x,
                         self.labels_placeholder: y,
                         self.dropout_placeholder: dp}
            state, loss, _ = session.run([self.final_state, self.calculate_loss, train_op], feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\rstep {} : pp = {}'.format(
                    step, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))


def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
    state = model.initial_state.eval()
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    for i in range(stop_length):
        feed_dict = {model.input_placeholder: [tokens[-1:]],
                     model.initial_state: state,
                     model.dropout_placeholder: 1}
        state, pred = session.run([model.final_state, model.predictions[-1]], feed_dict=feed_dict)
        genword_index = sample(pred[0], temp)
        tokens.append(genword_index)
        if model.vocab.decode(genword_index) in stop_tokens:
            break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output


def generate_sentence(session, model, config, *args, **kwargs):
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)


def test_RNNLM():
    config = Config()

    # gen_config
    gen_config = deepcopy(config)
    gen_config.batch_size = 1
    gen_config.num_steps = 1
    with tf.variable_scope('RNNLM') as scope:
        # model to train
        model = RNNLM(config)
        model.train_step = model.add_training_op(model.calculate_loss)
        # This instructs gen_model to reuse the same variables as the model above
        scope.reuse_variables()
        gen_model = RNNLM(gen_config)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
        session.run(init)
        start = time.time()
        for epoch in range(config.max_epochs):
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step)
            valid_pp = model.run_epoch(session, model.encoded_valid)
            print('Epoch {},train pp {}, valid pp {}'.format(epoch, train_pp, valid_pp))
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, '../models/ptb_rnnlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

        saver.restore(session, '../models/ptb_rnnlm.weights')
        test_pp = model.run_epoch(session, model.encoded_test)
        print('=-=' * 5)
        print('Test perplexity: {}'.format(test_pp))

        sentence = generate_sentence(session, gen_model, gen_config, starting_text="you have", temp=1.0)
        print(" ".join(sentence))
        sentence = generate_sentence(session, gen_model, gen_config, starting_text="this is", temp=1.0)
        print(" ".join(sentence))

def test_generate():
    path = '../models/ptb_rnnlm.weights'
    gen_config = Config()
    gen_config.batch_size = 1
    gen_config.num_steps = 1
    gen_model = RNNLM(gen_config)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph(path + '.meta')
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        sentence = generate_sentence(session, gen_model, gen_config, starting_text="you have", temp=1.0)
        print(" ".join(sentence))


if __name__ == '__main__':
    # test_RNNLM()
    test_generate()
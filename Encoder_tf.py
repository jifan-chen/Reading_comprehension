import tensorflow as tf
from tensorflow.contrib import rnn


class RNN_encoder():

    def __init__(self, hidden_size, vscope, bidirectional):
        self.hidden_size = hidden_size
        self.vscope = vscope
        self.bidirectional = bidirectional

    def encode( self, x, sequence_length):
        # Forward direction cell
        if self.bidirectional:
            with tf.variable_scope(self.vscope):
                with tf.variable_scope('forward_rnn'):
                    lstm_fw_cell = rnn.GRUCell(self.hidden_size)
                # Backward direction cell
                with tf.variable_scope('backward_rnn'):
                    lstm_bw_cell = rnn.GRUCell(self.hidden_size)


                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                         sequence_length=sequence_length,
                                                                         dtype=tf.float32)
                ht = tf.concat([output_states[0], output_states[1]], 1)
                rnn_sequence = tf.concat([outputs[0], outputs[1]], 2)
        else:
            with tf.variable_scope(self.vscope):
                lstm_cell = rnn.GRUCell(self.hidden_size)
                outputs, output_states = tf.nn.dynamic_rnn(lstm_cell, x,sequence_length=sequence_length,dtype=tf.float32)
                ht = output_states
                rnn_sequence = outputs
        return ht, rnn_sequence
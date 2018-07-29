# -*- coding: utf-8 -*-
class PointerNetwork(object):
    def __init__(self):
        pass

    def forward(self, inputs):
        """
        :param inputs: inputs sequence, [bz, seq_len, embedding_dim]
        :return: outputs
                           pointers
        """
        pass

    def encoder(self, inputs):
        """
        use LSTM to encode the input sequence.
        :param inputs: inputs seq, [bz, seq_len, embedding_dim]
        :return:  encoder_outputs:
                             encoder_hidden:
        """
        pass

    def decoder(self, embedded_inputs, decoder_input0,
                decoder_hidden0, encoder_outputs):
        """
        decoder  to get the output and pointers.
        :param embedded_inputs:
        :param decoder_input0:
        :param decoder_hidden0:
        :param encoder_outputs:
        :return:
                 outputs:
                 pointers:
        """
        pass


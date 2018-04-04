from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Bidirectional, LSTM, Dropout, Dense, Concatenate
from keras.engine import Model
from keras.initializers import RandomUniform, glorot_uniform, Constant, uniform
from keras.optimizers import SGD
from keras_contrib.layers import CRF
import numpy as np
import config
import utils


def build_model(data):
    # inputs
    chars_input = Input([data.max_sen_len, data.max_word_len], dtype='int32')
    words_input = Input([data.max_sen_len, ], dtype='int32')
    pos_input = Input([data.max_sen_len, ], dtype='int32')

    # embeddings
    scale = np.sqrt(3.0 / config.char_embed_dim)
    chars = Embedding(data.char_alphabet_size+2, config.char_embed_dim, embeddings_initializer=RandomUniform(-scale, scale), mask_zero=True)(chars_input)
    words = Embedding(*data.word_embeddings.shape, weights=[data.word_embeddings], trainable=False)(words_input)
    pos = Embedding(data.pos_alphabet_size+2, data.pos_alphabet_size, embeddings_initializer='identity', mask_zero=True)(pos_input)

    if config.dropout is not False:
        chars = Dropout(config.dropout)(chars)

    # char-level word feature
    cnn = Conv1D(config.num_filters, config.conv_window, padding='same', activation='tanh')(chars)
    pool = GlobalMaxPool1D()(cnn)

    # word representation
    incoming = Concatenate()([words, pos, pool])

    if config.dropout is not False:
        incoming = Dropout()(incoming)

    # Bi-LSTM
    bi_lstm = Bidirectional(LSTM(
        config.num_units,
        kernel_initializer=glorot_uniform(),
        recurrent_initializer=uniform(-0.1, 0.1),
        bias_initializer=Constant(1.),
        recurrent_activation='tanh'
    ))(incoming)

    if config.dropout is not False:
        bi_lstm = Dropout()(bi_lstm)

    # CRF
    crf = CRF(data.num_labels)(bi_lstm)

    model = Model(inputs=[chars_input, words_input, pos_input], outputs=[crf])
    optimizer = SGD(lr=config.learning_rate, momentum=config.momentum)
    model.compile(loss=crf.loss_function, metrics=[crf.accuracy], optimizer=optimizer)
    model.summary()
    return model


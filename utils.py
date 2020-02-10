import re

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._continuous_distns import _norm_pdf
from scipy.stats._distn_infrastructure import rv_sample


def tf_lookback(series, shift):
    """
    Dhifts the series forward by shift to allow to see older data.
    """
    s_len = K.shape(series)[0]
    # copy first elements, there are no elements that would be shifted to them (can be set to zero also)
    # (the last elements are not needed to predict anything and cannot be rolled to the beginning)
    return K.concatenate([series[:shift, ...],
                          series[:s_len - shift, ...]
                          ], axis=0)


def get_date_infos(dateseries):
    """
    Takes a datetime series and extracts day and month information.
    """
    return np.stack([dateseries.dayofweek.to_numpy(), dateseries.month.to_numpy(), dateseries.day.to_numpy()], axis=-1)


# get_date_infos(data_normalized.loc['2017-1-3':'2018-12-28'].index).shape


def npindices_to_onehot(input):
    """
    Takes indices array and produces onehot encoded numpy array.
    """
    input = input.astype(int)
    onehot = np.zeros((input.size, input.max() + 1), dtype=int)
    onehot[np.arange(input.size), input] = 1
    return onehot


def get_date_infos_discrete(dateseries):
    return np.concatenate(
        [npindices_to_onehot(dateseries.dayofweek.to_numpy()), npindices_to_onehot(dateseries.month.to_numpy())],
        axis=-1)


# get_date_infos_discrete(data_normalized.loc['2017-1-3':'2018-12-28'].index).shape


def vectorize(text, embeddings_model):
    """
    Vectorize given text by gensim embeddings model.
    """
    text = text.lower()
    if text in embeddings_model.vocab:
        return embeddings_model[text]
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    if text in embeddings_model.vocab:
        return embeddings_model[text]
    return np.zeros(embeddings_model.vector_size)


def np_pad_to_size(arrays, minsizes=None, default=0, dtype=None):
    """
    All arrays will be padded to have the same size.
    default will get copied by its shape.
    """
    dt_representant = None
    for item in arrays:
        if item is not None:
            dt_representant = item
    assert dt_representant is not None, "provide at least one numpy array in the list"
    assert all([item is None or item.ndim == dt_representant.ndim for item in
                arrays]), "arrays need to be at least the same ndim"
    
    shape = [max([item.shape[i] for item in arrays if item is not None])
             for i in range(dt_representant.ndim)]
    shape = [len(arrays)] + shape
    
    if minsizes is not None:
        # defined from the back!
        for i in range(min(len(minsizes), len(shape))):
            if minsizes[-1 - i] is not None:
                shape[-1 - i] = max(shape[-1 - i], minsizes[-1 - i])
    
    padded_array = np.full(tuple(shape), default, dtype=dtype if dtype is not None else dt_representant.dtype)
    # Here we put all the data to the beginning and leave the padded at te end.
    # Another options are to put it at random position, end, or overflow...
    for i, item in enumerate(arrays):
        if item is not None:
            padded_array[i][tuple([slice(0, n) for n in item.shape])] = item
    
    return padded_array


class SinCosPositionalEmbedding(Layer):
    """
    As in https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf, adapted to our needs

    If list of from_inputs_features not provided, takes a range of the input sequence length.
    If from_inputs_features is provided, takes the indices from the last dimension in from_inputs_features
     to be the positions to be embedded.

    Just to note - what we want to accomplish: for i 0...2*self.embed_dim to have these features:
    sins = tf.sin(pos /tf.pow(self.pos_divisor, 2.0 * i / self.embed_dim))
    coss = tf.cos(pos /tf.pow(self.pos_divisor, 2.0 * i / self.embed_dim))
    """
    
    def __init__(self, embed_dim, from_inputs_features=None, pos_divisor=10000, keep_ndim=True, fix_range=None,
                 embeddings=['sin', 'cos'], **kwargs):
        """
        embed_dim: Te output embedding will have embed_dim floats for sin and cos (separately).
        from_inputs_features: If not specified, will use range(of the length of the input sequence) to generate
            (integer) positions that will be embedded by sins and coss.
            If specified, it needs to be a list of coordinates to the last dimension of the input vector,
             which will be taken as inputs into the positional ebedding.
             Then the output size will be len(from_inputs_features)*embed_dim*len(embeddings)
             Has no effect when fix<-range is set.
        pos_divisor: the division constant in the calculation.
        keep_ndim: if True, the output will have all embedded features concatenated/flattened into one dimension and so
            the input dimensions number is preserved.
        fix_range: if set, will produce a sequence of a fixed range (does not read from sequence length)
            and also disables from_inputs_features.
        embeddings: a list of 'sin', 'cos', 'lin' functions to be applied
        """
        Layer.__init__(self, **kwargs)
        self.pos_divisor = pos_divisor
        self.embed_dim = embed_dim
        self.keep_ndim = keep_ndim
        self.from_inputs_features = from_inputs_features
        self.fix_range = fix_range
        self.embeddings = embeddings
    
    def get_config(self):
        config = {'pos_divisor': self.pos_divisor,
                  'embed_dim': self.embed_dim,
                  'keep_ndim': self.keep_ndim,
                  'from_inputs_features': self.from_inputs_features,
                  'fix_range': self.fix_range,
                  'embeddings': self.embeddings,
                  }
        base_config = super(SinCosPositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        self.built_multipliers = tf.expand_dims(
            tf.constant([pow(self.pos_divisor, -2.0 * i / self.embed_dim)
                         for i in range(self.embed_dim)]), 0)
        # ^^ [1, self.embed_dim]
        self.built = True
    
    def compute_output_shape(self, input_shape):
        # assumes input in the shape of [..., batches, sequence_len, features]
        features_to_position = len(self.from_inputs_features) if self.from_inputs_features is not None else 1
        embeddings = [features_to_position, self.embed_dim, len(self.embeddings)]
        if self.keep_ndim:
            return tuple(list(input_shape[:-1]) + [np.prod(embeddings)])
        else:
            return tuple(list(input_shape[:-1]) + embeddings)
    
    def call(self, inputs, training=None):
        # assumes input in the shape of [..., batches, sequence_len, features]
        if self.from_inputs_features:
            n_positions = len(self.from_inputs_features)
            positions = tf.gather(inputs, self.from_inputs_features, axis=-1)
        else:
            if self.fix_range is not None:
                seq_len = self.fix_range
            else:
                seq_len = tf.shape(inputs)[-2]
            n_positions = 1
            positions = tf.expand_dims(tf.to_float(tf.range(seq_len)), -1)
        # features_to_position = len(self.from_inputs_features) if self.from_inputs_features is not None else 1
        # now positions is [..., batches, sequence_len, features_to_position]
        # now for each features_to_position, we will create positional embedding of self.embed_dim in sin and cos, so
        # totally 2*self.embed_dim
        # self.built_multipliers is [1, self.embed_dim]
        # we want to get [..., batches, sequence_len, features_to_position (, or x) self.embed_dim]
        # so we need to reshape so that we get positions into [..., batches, sequence_len, features_to_position, 1]
        
        shape_batches_and_sequence = tf.shape(positions)[:-1]  # [..., batches, sequence_len]
        to_mult_shape = tf.concat([shape_batches_and_sequence, tf.shape(self.built_multipliers)], axis=0)
        try:
            # newer tensorflow has this function
            broadcast_ = tf.broadcast_to(self.built_multipliers, to_mult_shape)
        except AttributeError:
            # if we do not have it, lets use this way:
            broadcast_ = self.built_multipliers + tf.zeros(dtype=self.built_multipliers.dtype, shape=to_mult_shape)
        
        positions_divided = tf.matmul(tf.expand_dims(positions, -1), broadcast_)
        # ^^ [..., batches, sequence_len, features_to_position, self.embed_dim]
        
        list_of_embeddidngs = []  # default will use [tf.sin(positions_divided), tf.cos(positions_divided)]
        for activation_str in self.embeddings:
            act_to_tf = {'sin': tf.sin, 'cos': tf.cos, 'lin': lambda x: x}
            tf_activation = act_to_tf[activation_str]
            list_of_embeddidngs.append(tf_activation(positions_divided))
        
        positions_embedded = tf.concat(list_of_embeddidngs, axis=-1)
        # ^^ [..., batches, sequence_len, features_to_position, self.embed_dim, 2]
        
        if self.keep_ndim:
            # positions_embedded = tf.reshape(positions_embedded, tf.concat([tf.shape(inputs)[:-1], [-1]], axis=0))
            last_dim = self.embed_dim * n_positions * len(self.embeddings)
            positions_embedded = tf.reshape(positions_embedded, tf.concat([tf.shape(inputs)[:-1], [last_dim]], axis=0))
            # ^^ [..., batches, sequence_len, last_dim]
            # we could have specified last_dim to be = -1, but then keras might get confused somewhere
        
        return positions_embedded


class conditioned_continuous(rv_continuous):
    """
    Just pick a standart distribution based on discrete model and then apply it.
    """

    def __init__(self, discrete_d, continuous_cases, **kwargs):
        rv_continuous.__init__(self, **kwargs)
        assert isinstance(discrete_d, rv_sample)
        self.discrete_d = discrete_d
        self.continuous_cases = continuous_cases

    def _pdf(self, x, *args):
        accum = np.zeros((len(x)))
        for pos, prob in zip(self.discrete_d.xk, self.discrete_d.pk):
            accum += prob*self.continuous_cases[pos].pdf(x)
        return accum

    def _rvs(self, *args):
        sampled_d = self.discrete_d.rvs(*args, size=self._size, random_state=self._random_state)
        sampled = np.empty_like(sampled_d, dtype=np.float)
        for val in self.discrete_d.xk:
            vrvs = self.continuous_cases[val].rvs(sampled_d.shape, random_state=self._random_state)
            where = sampled_d == val
            sampled[where] = vrvs[where]
           
        return sampled

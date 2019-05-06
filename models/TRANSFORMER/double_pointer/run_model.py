"""
Experiment Notes:

This uses a transformer for the start pred and uses
a transformer for the end pred.

The transformers are NOT shared. They use their own weights.

"""


import os
from functools import partial

import tensorflow as tf
from backend.model_helpers import get_forget_bias, load_embeddings
from tensorflow import logging as logging
from tensorflow.python.client import device_lib
from tensorflow.python.layers.core import Dense
from training.train_pointer import train_model
from training.utils import gather_args

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# debugging
# from pdb import set_trace  # noqa

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def transform_to_range(input_val, min_value, max_value):
    scale = (max_value - min_value)
    rescaled = input_val * scale + min_value
    return rescaled


def layer_norm(input_tensor, epsilon=1e-6):
    mean, variance = tf.nn.moments(input_tensor, axes=[-1], keep_dims=True) # along the last dim -- the embedding dim
    output_tensor = (input_tensor - mean) / (variance + epsilon)
    return output_tensor

def reshape_transpose(tensor, batch_size, seq_length, num_heads, size_per_head):
    " rearrange the first two dims after the batch dim "
    # transpose key for the matmul (to move the sequences in to the matmul position with the heads)
    reshaped = tf.reshape(tensor, shape=[batch_size, seq_length, num_heads, size_per_head])  #16x200x8x64
    transposed = tf.transpose(reshaped, [0, 2, 1, 3])  #16x8x200x64
    return transposed

def init_kaiman_kernel(shape, name, prev_layer_size):
    kaiman = tf.random.normal(
        shape=shape,
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        seed=None,
        name='random_normal') * tf.sqrt(tf.constant(2, tf.float32) / tf.constant(prev_layer_size, tf.float32))
    return tf.get_variable(name=name, shape=shape, initializer=kaiman)

# input dim divisible by 4
def multihead_attention(transformer_input, batch_size, seq_length=200, num_heads=8, size_per_head=64):

    #transformer_input = 16x200x512
    # kaiman_init = init_kaiman_kernel(shape=(transformer_input.shape[-1], num_heads * size_per_head), name='keyinit', prev_layer_size=transformer_input.shape[-2])
    init = tf.contrib.layers.xavier_initializer
    with tf.variable_scope('QKV_linear_projection', reuse=True):
        # Its a linear projection, bias is not specified
        key_layer   = tf.layers.Dense(num_heads * size_per_head, activation=tf.nn.relu, use_bias=True, name='K', kernel_initializer=init())  #512x512
        query_layer = tf.layers.Dense(num_heads * size_per_head, activation=tf.nn.relu, use_bias=True, name='Q', kernel_initializer=init()) #512x512
        value_layer = tf.layers.Dense(num_heads * size_per_head, activation=tf.nn.relu, use_bias=True, name='V', kernel_initializer=init()) #512x512

    key   = key_layer(transformer_input)   # create key   16x200x512
    query = query_layer(transformer_input) # create query 16x200x512
    value = value_layer(transformer_input) # create value 16x200x512

    tf.summary.histogram('key', key)
    tf.summary.histogram('query', query)
    tf.summary.histogram('value', value)

    key_dim = tf.shape(key)[-1]

    # reshaping and transposing allows for the raw scores to be computed across all heads
    # without the need for splitting. Remember -- in a world with 1 head we would just use the K,Q,V directly
    # and not worry about dealing with replicates of itself. But to do the same operation in a single go
    # with mulitple attention heads (copies of the input dotted with mulitple key, query, and value matrices)
    # we can reshape the key, query, AND value to split these outputs to indvidual head size copies, so that they
    # can be dotted together.
    reshape_transpose_ = partial(
        reshape_transpose,
        batch_size=batch_size, seq_length=seq_length, num_heads=num_heads, size_per_head=size_per_head
    )
    K = reshape_transpose_(key)  # 16x8x200x64
    Q = reshape_transpose_(query) # 16x8x200x64
    V = reshape_transpose_(value) # 16x8x200x64

    raw_scores = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))   #16x8x200x64 dot 16x8x64x200 => 16x8x200x200

    scaling_factor = tf.sqrt(tf.to_float(key_dim))
    scaled_raw_scores = tf.divide(tf.to_float(raw_scores), scaling_factor) # 16x8x200x200

    attention_probs = tf.nn.softmax(scaled_raw_scores, name='attention_probs')                     # 16x8x200x200
    masked_attention_probs = tf.linalg.LinearOperatorLowerTriangular(attention_probs).to_dense()

    self_attended = tf.matmul(tf.cast(masked_attention_probs, tf.float64), tf.cast(V, tf.float64))     # 16x8x200x200 dot 16x8x200x64 => 16x8x200x64

    tf.summary.histogram('attention_probs', masked_attention_probs)

    self_attended = tf.transpose(self_attended, [0, 2, 1, 3])  # 16x200x8x64
    self_attended = tf.reshape(self_attended, [batch_size, seq_length, size_per_head*num_heads])  # 16x200x512
    self_attended = tf.nn.dropout(self_attended, keep_prob=0.85)

    # add and norm residuals
    residuals1 = tf.add(self_attended, tf.cast(transformer_input, tf.float64), name='residuals1')  #16x200x512
    normalized_outputs1 = layer_norm(residuals1)

    # fully connected layer w/ dropout
    fcl_out = tf.layers.dense(normalized_outputs1, num_heads * size_per_head, activation=tf.nn.relu, use_bias=True, name='fcl1', kernel_initializer=init())
    fcl_out = tf.nn.dropout(fcl_out, keep_prob=0.5)

    # add and norm residuals a second time
    residuals2 = tf.add(normalized_outputs1, fcl_out, name='residuals2')
    normalized_outputs = layer_norm(residuals2)
    tf.summary.histogram('normalized_outputs', normalized_outputs)
    return normalized_outputs   # 16x200x512 <- average these 200 axis 1

def data_pipeline_calls(
    features,
    params,
    mode,
    GPU
    ):

    init = tf.initializers.truncated_normal(0.0, 0.01)

    # if you want to train the embeds from scratch
    # embedding_vectors = tf.get_variable(name='embedding_vectors', shape=(params['vocab_size'], 512), initializer=init)
    # embedded_input = tf.nn.embedding_lookup(embedding_vectors, features['encoder_inputs'])

    # If you don't want to train the embeddings:
    lookup_table, emb_vectors = load_embeddings(params['embedding_vectors'], params['vocab'])
    embedded_input = tf.nn.embedding_lookup(emb_vectors, features['encoder_inputs'])

    forget_bias = get_forget_bias(params, mode)
    with tf.device(GPU['sidekick']):
        # high_dim_embedding_vecs = tf.layers.dense(embedded_input, units=512, activation=tf.nn.relu)

        positional_embeddings = tf.get_variable('position_embedding', shape=(params['input_max_length'], 50))

        positions = tf.range(params['input_max_length'])
        positions = tf.reshape(tf.tile(positions, [params['batch_size']]), (-1, params['input_max_length']))
        position_embeddings = tf.cast(tf.nn.embedding_lookup(positional_embeddings, positions), tf.float32)


    transformer_input = tf.add(embedded_input, position_embeddings)
    transformer_input = tf.nn.dropout(transformer_input, keep_prob=0.5)

    return transformer_input


def model_fn(features, labels, mode, params):
    # particular to this project
    word2index = params['word2index']

    GPUs = get_available_gpus()
    CPUs = get_available_cpus()

    GPU = {
        'titan': GPUs[0],
        'sidekick': GPUs[1]}
    CPU = {
        'main_cpu': CPUs[0]
    }

    transformer_size = 256
    num_heads = 8

    size_per_head = int(transformer_size / num_heads)
    assert int(size_per_head * num_heads) == transformer_size
    init = tf.contrib.layers.xavier_initializer


    transformer_input = data_pipeline_calls(features, params, mode, GPU)

    # increase dimensions
    transformer_input = tf.layers.dense(transformer_input, activation=tf.nn.relu, units=transformer_size, kernel_initializer=init())
    tf.summary.histogram('transformer_input', transformer_input)

    with tf.device(GPU['titan']):

        with tf.variable_scope('pointer_1'):
            with tf.variable_scope('pointer_layer1'):
                start_transformed = multihead_attention(transformer_input, params['batch_size'], seq_length=params['input_max_length'], num_heads=num_heads, size_per_head=size_per_head)
            with tf.variable_scope('pointer_layer2'):
                start_transformed = multihead_attention(start_transformed, params['batch_size'], seq_length=params['input_max_length'], num_heads=num_heads, size_per_head=size_per_head)

        with tf.variable_scope('pointer_2'):
            with tf.variable_scope('end_transformer_layer_1'):
                end_transformed = multihead_attention(transformer_input, params['batch_size'], seq_length=params['input_max_length'], num_heads=num_heads, size_per_head=size_per_head)
            with tf.variable_scope('end_transformer_layer_2'):
                end_transformed = multihead_attention(end_transformed, params['batch_size'], seq_length=params['input_max_length'], num_heads=num_heads, size_per_head=size_per_head)

        start_flat = tf.reduce_mean(start_transformed, axis=1, name='encoded_text_start')
        start_flat = tf.layers.flatten(start_transformed, name='start_flatten')

        end_flat = tf.reduce_mean(end_transformed, axis=1, name='encoded_text_end')
        end_flat = tf.layers.flatten(end_transformed, name='end_flatten')

        # start pred
        start_hidden = tf.layers.dense(start_flat, units=params['input_max_length'])
        start_predictions = tf.layers.dense(start_hidden, units=1, activation=tf.nn.sigmoid, kernel_initializer=init())
        start_predictions_transformed = transform_to_range(start_predictions, min_value=0, max_value=params['input_max_length'])

        end_input = tf.concat((start_predictions, end_flat), 1, name='end_input')
        end_hidden = tf.layers.dense(end_input, units=params['input_max_length'])
        end_predictions = tf.layers.dense(end_hidden, activation=tf.nn.sigmoid, use_bias=True, units=1)
        end_predictions_transformed = transform_to_range(end_predictions, min_value=0, max_value=params['input_max_length'])


    if mode == tf.estimator.ModeKeys.PREDICT:

        starts = tf.to_int32(start_predictions_transformed)
        ends = tf.to_int32(end_predictions_transformed)

        predictions = {'question_starts': starts, 'question_ends': ends}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    starts = labels['starts']

    # compute losses
    question_start_labels = tf.reshape(tf.to_float(labels['starts']), ( -1, 1))
    question_end_labels = tf.reshape(tf.to_float(labels['stops']), (-1, 1))

    start_loss = tf.losses.mean_squared_error(labels=question_start_labels, predictions=start_predictions_transformed)
    end_loss = tf.losses.mean_squared_error(labels=question_end_labels, predictions=end_predictions_transformed)

    order_penalty = tf.cast(
        tf.square(
            tf.cast(
                tf.nn.relu(start_predictions_transformed - end_predictions_transformed) + 1,
                tf.float32)
        ), tf.float32
    )

    combined_loss = start_loss + end_loss + tf.reduce_mean(order_penalty)

    tf.summary.scalar('start_loss', start_loss)
    tf.summary.scalar('end_loss', end_loss)
    tf.summary.scalar('penalty_loss', tf.reduce_mean(order_penalty))

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'combined_loss': combined_loss
            }
        return tf.estimator.EstimatorSpec(mode, loss=combined_loss, eval_metric_ops=metrics)


    global_step = tf.train.get_global_step()

    learning_rate = 0.0005
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)
    gradients = optimizer.compute_gradients(combined_loss)

    grad_summ_op = tf.summary.merge([tf.summary.histogram("{}-grad".format(g[1].name).replace(':', '_'), g[0]) for g in gradients])

    capped_gradients = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=combined_loss, train_op=train_op)


if __name__ == "__main__":
    """
    If we start from scratch, we'll make a new directory
     - if we name the directory, it takes the name, else, it is called 'test_directory'
    If we change hyperparameters, change the name of the output_dir.
    """
    args = gather_args()
    train_model(args, model_fn)

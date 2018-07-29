# -*- coding: utf-8 -*-
import paddle
import paddle.fluid as fluid

import numpy
from functools import partial
import math
import os
import sys

# params
EMBED_SIZE = 32  # word vector dimension
HIDDEN_SIZE = 256 # hidden layer dimension
N = 5  # train 5-gram
BATCH_SIZE = 5

# can use cpu or gpu
use_cuda = False

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)


def inference_program(is_sparse):
    """
    construct  5-gram neural language model.
    :param is_sparse:
    :return:
    """
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.layers.embedding(
        input=fourth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(
        input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word


def train_program(is_sparse):
    """
    define the train program.
    :param is_sparse:
    :return: cross-entropy loss.
    """
    # The declaration of 'next_word' must be after the invoking of inference_program,
    # or the data input order of train program would be [next_word, firstw, secondw,
    # thirdw, fourthw], which is not correct.
    predict_word = inference_program(is_sparse)
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    """optimizer function"""
    return fluid.optimizer.AdagradOptimizer(
        learning_rate=3e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(8e-4))


def train(use_cuda, train_program, params_dirname):
    """main train function."""
    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(train_func=train_program,
                            optimizer_func=optimizer_func,
                            place=place)

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            outs = trainer.test(
                reader=test_reader,
                feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
            avg_cost = outs[0]

            if event.step % 10 == 0:
                print("Step %d: Average cost %f" % (event.step, avg_cost))
            if avg_cost < 5.8:
                trainer.save_params(params_dirname)
                trainer.stop()

            if math.isnan(avg_cost):
                sys.exit("got NaN loss, training failed")
    print("begin training....")
    trainer.train(
        reader=train_reader,
        num_epochs=1,
        feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'],
        event_handler=event_handler
    )


def infer(use_cuda, inference_program, params_dirname=None):
    """
    infer  use the trained model.
    :param use_cuda:
    :param inference_program:
    :param params_dirname:
    :return:
    """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    # Setup inputs by creating 4 LoDTensors representing 4 words. Here each word
    # is simply an index to look up for the corresponding word vector and hence
    # the shape of word (base_shape) should be [1]. The length-based level of
    # detail (lod) info of each LoDtensor should be [[1]] meaning there is only
    # one lod_level and there is only one sequence of one word on this level.
    # Note that lod info should be a list of lists.

    data1 = [[211]]  # 'among'
    data2 = [[6]]  # 'a'
    data3 = [[96]]  # 'group'
    data4 = [[4]]  # 'of'
    lod = [[1]]

    first_word = fluid.create_lod_tensor(data1, lod, place)
    second_word = fluid.create_lod_tensor(data2, lod, place)
    third_word = fluid.create_lod_tensor(data3, lod, place)
    fourth_word = fluid.create_lod_tensor(data4, lod, place)

    result = inferencer.infer(
        {
            'firstw': first_word,
            'secondw': second_word,
            'thirdw': third_word,
            'fourthw': fourth_word
        },
        return_numpy=False)

    print(numpy.array(result[0]))
    most_possible_word_index = numpy.argmax(result[0])
    print(most_possible_word_index)
    print([
        key for key, value in word_dict.iteritems()
        if value == most_possible_word_index
    ][0])


if __name__ == "__main__":
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        sys.exit("paddlepaddle not compiled with cuda")

    is_sparse = True
    params_dirname = "word2vec.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=partial(inference_program, is_sparse),
        params_dirname=params_dirname
    )
    # inference
    infer(use_cuda=use_cuda,
          inference_program=partial(inference_program, is_sparse),
          params_dirname=params_dirname)










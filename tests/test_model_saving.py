import pytest
import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.layers import Input
from keras import optimizers
from keras import objectives
from keras import metrics
from keras.utils.test_utils import keras_test
from keras.models import save_model, load_model, get_mxnet_model, save_mxnet_model


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_get_mxnet_model_pred():
    model = Sequential()
    model.add(Dense(8, input_shape=(32,)))
    model.add(Dense(4, input_shape=(8,)))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    X = np.random.random((8, 32))
    Y = np.random.random((8, 4))
    model.fit(X, Y, batch_size=8, nb_epoch=5)

    module, symbol = get_mxnet_model(model)

    import mxnet as mx

    assert type(symbol) is mx.symbol.Symbol
    assert type(module) is mx.module.BucketingModule


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_get_mxnet_model_train():
    model = Sequential()
    model.add(Dense(8, input_shape=(32,)))
    model.add(Dense(4, input_shape=(8,)))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    X = np.random.random((8, 32))
    Y = np.random.random((8, 4))
    model.fit(X, Y, batch_size=8, nb_epoch=5)

    module, symbol = get_mxnet_model(model, bucket='train')

    import mxnet as mx

    assert type(symbol) is mx.symbol.Symbol
    assert type(module) is mx.module.BucketingModule


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_get_mxnet_model_failures():
    # None type passed in
    with pytest.raises(AssertionError):
        get_mxnet_model(None)

    model = Sequential()
    model.add(Dense(8, input_shape=(32,)))
    model.add(Dense(4, input_shape=(8,)))

    # Model not compiled
    with pytest.raises(AssertionError):
        get_mxnet_model(model)

    # Wrong bucket passed in
    model.compile(loss='mean_squared_error', optimizer='sgd')
    with pytest.raises(ValueError):
        get_mxnet_model(model, bucket='random')

    # Model still not trained so no underlying bucket
    with pytest.raises(ValueError):
        get_mxnet_model(model, bucket='train')


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_sequential_lstm_mxnet_model_saving():
    maxlen = 80

    model = Sequential()
    model.add(Embedding(1000, 128, input_length=maxlen))
    model.add(LSTM(128, unroll=True))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Generate data for dummy network
    X = np.random.random((1000, 80))
    Y = np.random.random((1000, 128))
    model.fit(X, Y, batch_size=32, nb_epoch=2)

    model_prefix = 'test_lstm'
    data_names, data_shapes = save_mxnet_model(model, prefix=model_prefix, epoch=0)

    # Import with MXNet and try to perform inference
    import mxnet as mx

    X_dummy_for_pred = mx.nd.random.normal(shape=(1, 80))
    pred_keras = model.predict([X_dummy_for_pred.asnumpy()], batch_size=1)

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch=0)
    mod = mx.mod.Module(symbol=sym, data_names=[data_names[0]], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], (1, 80))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    data_iter = mx.io.NDArrayIter([X_dummy_for_pred], label=None, batch_size=1)
    pred_mxnet = mod.predict(data_iter)

    # Check if predictions made through mxnet and keras model are same
    assert_allclose(pred_mxnet.asnumpy(), pred_keras, rtol=1e-03)

    os.remove(model_prefix + "-symbol.json")
    os.remove(model_prefix + "-0000.params")


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_sequential_dense_mxnet_model_saving():
    model = Sequential()
    model.add(Dense(8, input_shape=(32,)))
    model.add(Dense(4, input_shape=(8,)))

    model.compile(loss='mean_squared_error', optimizer='sgd')

    # Generate data for dummy network
    X = np.random.random((8, 32))
    Y = np.random.random((8, 4))
    model.fit(X, Y, batch_size=8, nb_epoch=5)

    model_prefix = 'test_dense'
    data_names, data_shapes = save_mxnet_model(model, model_prefix, epoch=1)

    # Import with MXNet and try to perform inference
    import mxnet as mx

    X_dummy_for_pred = mx.nd.random.normal(shape=(4, 32))
    pred_keras = model.predict([X_dummy_for_pred.asnumpy()], batch_size=4)

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch=1)
    mod = mx.mod.Module(symbol=sym, data_names=[data_names[0]], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], (4, 32))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    data_iter = mx.io.NDArrayIter([X_dummy_for_pred], label=None, batch_size=1)
    pred_mxnet = mod.predict(data_iter)

    # Check if predictions made through mxnet and keras model are same
    assert_allclose(pred_mxnet.asnumpy(), pred_keras, rtol=1e-03)

    os.remove(model_prefix + "-symbol.json")
    os.remove(model_prefix + "-0001.params")


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_sequential_mxnet_model_not_compiled():
    model = Sequential()
    model.add(Dense(8, input_shape=(32,)))
    model.add(Dense(4, input_shape=(8,)))

    model_prefix = 'test_dense'
    with pytest.raises(AssertionError):
        save_mxnet_model(model, model_prefix, epoch=0)


@keras_test
@pytest.mark.skipif((K.backend() != 'mxnet'), reason='Supported for MXNet backend only')
def test_model_dense_mxnet_model_saving():
    input = Input(shape=(32,))
    hidden_output = Dense(8, input_shape=(32,))(input)
    model = Model([input], [hidden_output])
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # Generate data for dummy network
    X = np.random.random((1, 32))
    Y = np.random.random((1, 8))
    model.fit(X, Y, batch_size=1, nb_epoch=2)

    model_prefix = 'test_model_api_dense'
    data_names, data_shapes = save_mxnet_model(model, model_prefix, epoch=0)

    # Import with MXNet and try to perform inference
    import mxnet as mx

    X_dummy_for_pred = mx.nd.random.normal(shape=(1, 32))
    pred_keras = model.predict([X_dummy_for_pred.asnumpy()], batch_size=1)

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch=0)
    mod = mx.mod.Module(symbol=sym, data_names=[data_names[0]], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], (1, 32))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    data_iter = mx.io.NDArrayIter([X_dummy_for_pred], label=None, batch_size=1)
    pred_mxnet = mod.predict(data_iter)

    # Check if predictions made through mxnet and keras model are same
    assert_allclose(pred_mxnet.asnumpy(), pred_keras, rtol=1e-03)

    os.remove(model_prefix + "-symbol.json")
    os.remove(model_prefix + "-0000.params")


@keras_test
@pytest.mark.skip(reason="Currently optimizer state is not preserved for mxnet backend.")
def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=objectives.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    new_model = load_model(fname)
    os.remove(fname)

    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test that new updates are the same with both models
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    new_model.train_on_batch(x, y)
    out = model.predict(x)
    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_sequential_model_saving_2():
    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = objectives.mse
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(Dense(3))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname,
                       custom_objects={'custom_opt': custom_opt,
                                       'custom_loss': custom_loss})
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_fuctional_model_saving():
    input = Input(shape=(3,))
    x = Dense(2)(input)
    output = Dense(3)(x)

    model = Model(input, output)
    model.compile(loss=objectives.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


@keras_test
def test_saving_without_compilation():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
def test_saving_right_after_compilation():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
    model.model._make_train_function()

    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)
    model = load_model(fname)
    os.remove(fname)


@keras_test
def test_loading_weights_by_name():
    """
    test loading model weights by name on:
        - sequential model
    """

    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = objectives.mse

    # sequential model
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="rick"))
    model.add(Dense(3, name="morty"))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    old_weights = [layer.get_weights() for layer in model.layers]
    _, fname = tempfile.mkstemp('.h5')

    model.save_weights(fname)

    # delete and recreate model
    del(model)
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="rick"))
    model.add(Dense(3, name="morty"))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    # load weights from first model
    model.load_weights(fname, by_name=True)
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)
    for i in range(len(model.layers)):
        new_weights = model.layers[i].get_weights()
        for j in range(len(new_weights)):
            assert_allclose(old_weights[i][j], new_weights[j], atol=1e-05)


@keras_test
def test_loading_weights_by_name_2():
    """
    test loading model weights by name on:
        - both sequential and functional api models
        - different architecture with shared names
    """

    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = objectives.mse

    # sequential model
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="rick"))
    model.add(Dense(3, name="morty"))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    old_weights = [layer.get_weights() for layer in model.layers]
    _, fname = tempfile.mkstemp('.h5')

    model.save_weights(fname)

    # delete and recreate model using Functional API
    del(model)
    data = Input(shape=(3,))
    rick = Dense(2, name="rick")(data)
    jerry = Dense(3, name="jerry")(rick)  # add 2 layers (but maintain shapes)
    jessica = Dense(2, name="jessica")(jerry)
    morty = Dense(3, name="morty")(jessica)

    model = Model(input=[data], output=[morty])
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    # load weights from first model
    model.load_weights(fname, by_name=True)
    os.remove(fname)

    out2 = model.predict(x)
    assert np.max(np.abs(out - out2)) > 1e-05

    rick = model.layers[1].get_weights()
    jerry = model.layers[2].get_weights()
    jessica = model.layers[3].get_weights()
    morty = model.layers[4].get_weights()

    assert_allclose(old_weights[0][0], rick[0], atol=1e-05)
    assert_allclose(old_weights[0][1], rick[1], atol=1e-05)
    assert_allclose(old_weights[1][0], morty[0], atol=1e-05)
    assert_allclose(old_weights[1][1], morty[1], atol=1e-05)
    assert_allclose(np.zeros_like(jerry[1]), jerry[1])  # biases init to 0
    assert_allclose(np.zeros_like(jessica[1]), jessica[1])  # biases init to 0


# a function to be called from the Lambda layer
def square_fn(x):
    return x * x


@keras_test
def test_saving_lambda_custom_objects():
    input = Input(shape=(3,))
    x = Lambda(lambda x: square_fn(x), output_shape=(3,))(input)
    output = Dense(3)(x)

    model = Model(input, output)
    model.compile(loss=objectives.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname, custom_objects={'square_fn': square_fn})
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__])

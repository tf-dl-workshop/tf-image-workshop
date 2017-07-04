from utils import *
from tensorflow.contrib import learn
from tensorflow.contrib.learn import *
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, params):
    """
    Model function for CNN
    :param features: images features with shape (batch_size, height, width, channels)
    :param labels: images category with shape (batch_size)
    :param mode: Specifies if this training, evaluation or
                 prediction. See `model_fn_lib.ModeKey`
    :param params: dict of hyperparameters
    :return: predictions, loss, train_op, Optional(eval_op). See `model_fn_lib.ModelFnOps`
    """

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool_flat = tf.reshape(pool3, [-1, 32 * 32 * 64])
    dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=5)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5, name="onehot")
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            summaries=[
                "learning_rate",
                "loss",
                "gradients",
                "gradient_norm",
            ])
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops={'dense': dense})


def feature_engineering_fn(features, labels):
    """
    feature_engineering_fn: Feature engineering function. Takes features and
                              labels which are the output of `input_fn` and
                              returns features and labels which will be fed
                              into `model_fn`
    """

    features = tf.to_float(features)

    # Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation

    # Example
    # Subtract off the mean and divide by the variance of the pixels.
    features = resnetpreprocessing(features)

    return features, labels


if __name__ == '__main__':
    cnn_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="_model/test_new2",
        config=RunConfig(save_summary_steps=10, keep_checkpoint_max=2, save_checkpoints_secs=30),
        feature_engineering_fn=feature_engineering_fn)

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    train_input_fn = read_img(data_dir='_data/train', batch_size=32, shuffle=True)
    monitor_input_fn = read_img(data_dir='_data/validate', batch_size=128, shuffle=False)
    validate_input_fn = read_img(data_dir='_data/test', batch_size=617, shuffle=False)

    validation_monitor = monitors.ValidationMonitor(input_fn=monitor_input_fn,
                                                    eval_steps=10,
                                                    every_n_steps=25,
                                                    metrics=metrics,
                                                    name='validation')

    cnn_classifier.fit(input_fn=train_input_fn, steps=1, monitors=[validation_monitor])

    # Evaluate the _model and print results
    eval_results = cnn_classifier.evaluate(input_fn=validate_input_fn, metrics=metrics, steps=1)
    np.save(os.getcwd() + '/embedding.npy', eval_results['dense'])

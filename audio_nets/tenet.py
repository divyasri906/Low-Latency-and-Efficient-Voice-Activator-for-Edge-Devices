
import tensorflow as tf
import tf_slim as slim


def TENet_arg_scope(is_training, weight_decay=0.00004, keep_prob=0.8):
    """
    TensorFlow 2.x compatible TENet argument scope.

    We keep tf_slim for convolution and other layers, but do NOT use
    slim.batch_norm because tf_slim's BatchNorm implementation depends
    on the removed legacy_tf_layers API with TensorFlow 2.21/Keras 3.
    """

    with slim.arg_scope(
        [slim.conv2d, slim.separable_convolution2d],
        activation_fn=tf.nn.relu,
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        biases_initializer=None,
        padding="SAME",
    ):
        with slim.arg_scope(
            [slim.dropout],
            keep_prob=keep_prob,
            is_training=is_training
        ) as scope:
            return scope


def _batch_norm(x, is_training, name):
    """
    TensorFlow/Keras BatchNormalization replacement for
    tf_slim.batch_norm.
    Explicitly registers moving_mean and moving_variance update ops to
    GraphKeys.UPDATE_OPS for TF1.x graph execution.
    """

    layer = tf.keras.layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001,
        name=name,
        trainable=True,
    )

    g = tf.compat.v1.get_default_graph()
    n_before = len(g.get_operations())
    y = layer(x, training=is_training)

    if is_training:
        if hasattr(layer, "updates") and layer.updates:
            for op in layer.updates:
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, op)
        else:
            # In Keras 3 (TensorFlow 2.16+), layer.updates is removed.
            # We match the moving_mean and moving_variance AssignVariableOps
            # directly by the layer's non_trainable_weights resource handles.
            handles = {w.value.handle for w in layer.non_trainable_weights}
            for op in g.get_operations()[n_before:]:
                if op.type == "AssignVariableOp" and op.inputs and op.inputs[0] in handles:
                    if not op.name.endswith("/Assign"):
                        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, op)

    return y


def _conv2d(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    scope=None,
    activation_fn=tf.nn.relu,
    is_training=True,
):
    """
    tf_slim convolution followed by TensorFlow BatchNormalization.

    This avoids tf_slim's legacy BatchNorm implementation.
    """

    net = slim.conv2d(
        inputs,
        num_outputs=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        activation_fn=None,
        normalizer_fn=None,
        scope=scope,
    )

    net = _batch_norm(
        net,
        is_training=is_training,
        name=f"{scope}_bn" if scope else None,
    )

    if activation_fn is not None:
        net = activation_fn(net)

    return net


def _separable_conv2d(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    depth_multiplier=1,
    scope=None,
    activation_fn=tf.nn.relu,
    is_training=True,
):
    """
    tf_slim separable convolution followed by TensorFlow
    BatchNormalization.
    """

    net = slim.separable_convolution2d(
        inputs,
        num_outputs=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        depth_multiplier=depth_multiplier,
        activation_fn=None,
        normalizer_fn=None,
        scope=scope,
    )

    net = _batch_norm(
        net,
        is_training=is_training,
        name=f"{scope}_bn" if scope else None,
    )

    if activation_fn is not None:
        net = activation_fn(net)

    return net


def tenet(
    inputs,
    num_classes,
    n_channels,
    n_strides,
    n_ratios,
    n_layers,
    kernel_list,
    scope,
    is_training=True,
):
    """
    TENet network.

    Architecture is kept the same as the original implementation.
    Only the BatchNormalization implementation is changed for
    TensorFlow 2.21 / Keras 3 compatibility.
    """

    L = inputs.shape[1]
    C = inputs.shape[2]

    assert len(n_channels) == len(n_strides) + 1

    with tf.compat.v1.variable_scope(scope):

        # [N, L, C] -> [N, L, 1, C]
        inputs = tf.reshape(
            inputs,
            [-1, L, 1, C]
        )

        first_conv_kernel = [3, 1]
        conv_kernel = [9, 1]

        # Initial convolution
        net = _conv2d(
            inputs,
            num_outputs=n_channels[0],
            kernel_size=first_conv_kernel,
            stride=1,
            scope="conv0",
            is_training=is_training,
        )

        n_channels = n_channels[1:]

        # TENet blocks
        for i, n in enumerate(n_channels):

            with tf.compat.v1.variable_scope(
                f"block{i}"
            ):

                expand_n = int(
                    n * n_ratios[i]
                )

                for j in range(n_layers[i]):

                    stride = (
                        n_strides[i]
                        if j == 0
                        else 1
                    )

                    # Residual/downsample path
                    if (
                        stride != 1
                        or net.shape[-1] != n
                    ):

                        layer_in = _conv2d(
                            net,
                            num_outputs=n,
                            activation_fn=None,
                            kernel_size=[1, 1],
                            stride=stride,
                            scope="down",
                            is_training=is_training,
                        )

                    else:

                        layer_in = net

                    # Pointwise expansion
                    net = _conv2d(
                        net,
                        num_outputs=expand_n,
                        kernel_size=[1, 1],
                        stride=1,
                        scope=f"pointwise_conv{j}_0",
                        is_training=is_training,
                    )

                    # Depthwise convolution
                    if kernel_list:

                        list_net = []

                        for k, kernel_size in enumerate(
                            kernel_list
                        ):

                            branch = _separable_conv2d(
                                net,
                                num_outputs=None,
                                activation_fn=None,
                                stride=stride,
                                depth_multiplier=1,
                                kernel_size=[
                                    kernel_size,
                                    1
                                ],
                                scope=f"depthwise_conv{j}_{k}",
                                is_training=is_training,
                            )

                            list_net.append(
                                branch
                            )

                        net = tf.add_n(
                            list_net
                        )

                        net = tf.nn.relu(
                            net
                        )

                    else:

                        net = _separable_conv2d(
                            net,
                            num_outputs=None,
                            stride=stride,
                            depth_multiplier=1,
                            kernel_size=conv_kernel,
                            scope=f"depthwise_conv{j}",
                            is_training=is_training,
                        )

                    # Pointwise projection
                    net = _conv2d(
                        net,
                        num_outputs=n,
                        activation_fn=None,
                        kernel_size=[1, 1],
                        stride=1,
                        scope=f"pointwise_conv{j}_1",
                        is_training=is_training,
                    )

                    # Residual connection
                    net += layer_in

        # Global average pooling
        net = slim.avg_pool2d(
            net,
            kernel_size=net.shape[1:3],
            stride=1,
            scope="avg_pool",
        )

        # Dropout
        net = slim.dropout(
            net,
            is_training=is_training,
        )

        # Final classifier
        logits = slim.conv2d(
            net,
            num_classes,
            1,
            activation_fn=None,
            normalizer_fn=None,
            scope="fc",
        )

        logits = tf.reshape(
            logits,
            shape=(-1, logits.shape[3]),
            name="squeeze_logit",
        )

    return logits


def TENet12(
    inputs,
    num_classes,
    kernel_list,
    scope,
    is_training=True,
):
    n_channels = [32] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [4] * 3

    if scope == "":
        if kernel_list:
            scope = "MTENet12"
        else:
            scope = "TENet12"

    return tenet(
        inputs,
        num_classes,
        n_channels,
        n_strides,
        n_ratios,
        n_layers,
        kernel_list,
        scope,
        is_training=is_training,
    )


def TENet6(
    inputs,
    num_classes,
    kernel_list,
    scope,
    is_training=True,
):
    n_channels = [32] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [2] * 3

    if scope == "":
        if kernel_list:
            scope = "MTENet6"
        else:
            scope = "TENet6"

    return tenet(
        inputs,
        num_classes,
        n_channels,
        n_strides,
        n_ratios,
        n_layers,
        kernel_list,
        scope,
        is_training=is_training,
    )


def TENet12Narrow(
    inputs,
    num_classes,
    kernel_list,
    scope,
    is_training=True,
):
    n_channels = [16] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [4] * 3

    if scope == "":
        if kernel_list:
            scope = "MTENet12Narrow"
        else:
            scope = "TENet12Narrow"

    return tenet(
        inputs,
        num_classes,
        n_channels,
        n_strides,
        n_ratios,
        n_layers,
        kernel_list,
        scope,
        is_training=is_training,
    )


def TENet6Narrow(
    inputs,
    num_classes,
    kernel_list,
    scope,
    is_training=True,
    width_multiplier=1.0,
):
    base_ch = int(round(16 * width_multiplier))
    n_channels = [base_ch] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [2] * 3

    if scope == "":
        if kernel_list:
            scope = "MTENet6Narrow"
        else:
            scope = "TENet6Narrow"

    return tenet(
        inputs,
        num_classes,
        n_channels,
        n_strides,
        n_ratios,
        n_layers,
        kernel_list,
        scope,
        is_training=is_training,
    )
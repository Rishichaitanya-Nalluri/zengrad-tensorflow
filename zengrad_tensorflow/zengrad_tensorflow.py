import tensorflow as tf
import keras

"""ZenGrad optimizer for TensorFlow / Keras 3 (TF >= 2.16).

    A_t   = A_{t-1} + g_t^2
    v_t   = mu * v_{t-1} + g_t
    lr_t  = lr / (log(A_t + 1) + eps)
    p    <- p * (1 - lr * wd) - lr_t * (g_t + mu * v_t)     (Nesterov momentum only)

Weight decay is decoupled and handled by the Keras base class
(variable *= 1 - lr * weight_decay), exactly as in the PyTorch version.
"""

@keras.saving.register_keras_serializable(package="zengrad")
class ZenGrad(keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.1,
        initial_accumulator_value=1.718281828,
        weight_decay=5e-4,
        epsilon=1e-8,
        momentum=0.9,
        name="ZenGrad",
        **kwargs,
    ):
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid accumulator value: {initial_accumulator_value}")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if momentum <= 0.0 or momentum >= 1.0:
            raise ValueError(
                f"Invalid momentum value: {momentum} (Nesterov momentum requires 0 < momentum < 1)"
            )

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._accumulators = [
            self.add_variable_from_reference(
                reference_variable=v,
                name="accumulator",
                initializer=keras.initializers.Constant(self.initial_accumulator_value),
            )
            for v in var_list
        ]
        self._momentum_buffers = [
            self.add_variable_from_reference(
                reference_variable=v,
                name="momentum_buffer",
                initializer="zeros",
            )
            for v in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        grad = tf.cast(gradient, variable.dtype)
        idx = self._get_variable_index(variable)

        # Accumulate squared gradients
        acc = self._accumulators[idx]
        acc.assign_add(tf.square(grad))

        # Effective (per-element) learning rate
        eps = tf.cast(self.epsilon, variable.dtype)
        effective_lr = lr / (tf.math.log1p(acc) + eps)

        # Nesterov momentum: v_t = mu * v_{t-1} + g_t ; step uses g_t + mu * v_t
        mu = tf.cast(self.momentum, variable.dtype)
        buf = self._momentum_buffers[idx]
        buf.assign(buf * mu + grad)
        update = grad + mu * buf

        variable.assign_sub(effective_lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
            }
        )
        return config

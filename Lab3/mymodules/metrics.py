import tensorflow as tf 

class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name="mean_absolute_error", **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight("total", initializer='zeros', shape=[2])
        self.count = self.add_weight("count", initializer="zeros")

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=0)
        self.total.assign_add(error)
        self.count.assign_add(1.)

        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result
        
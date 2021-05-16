import numpy as np
from tensorflow import GradientTape, losses, squeeze, reduce_sum, convert_to_tensor
from tensorflow.keras import Model as KerasModel, layers, optimizers
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import load_model


LEARNING_RATE = 0.001


class CustomKerasModel(KerasModel):

    def train_step(self, data):
        """
        Process one batch of training data
        todo: for sample weight see implementation in base class
        Have to override to multiply the one-hot encoded actions as a mask
        """
        states, target_q_values, actions_one_hot = data
        with GradientTape() as tape:
            qvalues = self(states, training=False)
            predicted_q_values = reduce_sum(qvalues * squeeze(actions_one_hot), axis=1)
            target_q_values_t = convert_to_tensor(target_q_values)
            loss = self.compiled_loss(target_q_values_t, predicted_q_values, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(target_q_values_t, predicted_q_values)
        return {m.name: m.result() for m in self.metrics}

    # def call(self, inputs, training=None, mask=None):
    #     pass
    #
    # def get_config(self):
    #     pass


class Model:
    def __init__(self, input_size, output_size, model_path=None,  name="SOAS_Model"):
        self._input_size = input_size
        self._output_size = output_size
        if model_path is None:
            self._model = self._create_model()
        else:
            self._model = load_model(model_path)

    def __call__(self, inputs, *args, **kwargs):
        return self._model(inputs, *args, **kwargs)

    def train_step(self, states, target_q_values, actions_one_hot):
        self._model.train_step((states, target_q_values, actions_one_hot))

    def copy_variables(self, other: 'Model'):
        self._model.set_weights(other._model.get_weights())

    def save(self, model_path):
        self._model.save(model_path)

    def save_weights(self, weights_path):
        self._model.save_weights(weights_path)

    def load_weights(self, weights_path):
        self._model.load_weights(weights_path)

    def _create_model(self):
        inputs = layers.Input(shape=self._input_size)
        x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                          activation='relu',
                          # default parameters result in He initialization that work better with relu activation
                          kernel_initializer=VarianceScaling())(inputs)
        x = layers.Conv2D(64, (2, 2), strides=(1, 1),
                          activation='relu',
                          # default parameters result in He initialization that work better with relu activation
                          kernel_initializer=VarianceScaling())(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu', kernel_initializer=VarianceScaling())(x)
        outputs = layers.Dense(np.prod(self._output_size), activation='relu', kernel_initializer=VarianceScaling())(x)
        model = CustomKerasModel(inputs, outputs)
        model.summary()

        model.compile(optimizer=optimizers.Adadelta(learning_rate=LEARNING_RATE),
                      loss=losses.Huber(),
                      metrics=['accuracy'])

        return model

    # def get_config(self):
    #     return {'input_size': self._input_size, 'output_size': self._output_size}
    #
    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     return cls(**config)

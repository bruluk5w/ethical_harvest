import numpy as np
from tensorflow import GradientTape, losses, squeeze, reduce_sum, convert_to_tensor
from tensorflow.keras import Model as KerasModel, layers, optimizers
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.python.ops.init_ops import GlorotUniform

LEARNING_RATE = 0.001
DENSE_MODEL = False


class CustomKerasModel(KerasModel):

    def train_step(self, data):
        """
        Process one batch of training data
        todo: for sample weight see implementation in base class
        Have to override to multiply the one-hot encoded actions as a mask
        """
        states, target_q_values, actions_one_hot, is_terminal = data
        with GradientTape() as tape:
            qvalues = self(states, training=False)
            predicted_q_values = reduce_sum(qvalues * squeeze(actions_one_hot), axis=1)
            target_q_values_t = convert_to_tensor(target_q_values)
            loss = self.compiled_loss(target_q_values_t, predicted_q_values, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(target_q_values_t, predicted_q_values)
        return {m.name: m.result() for m in self.metrics}


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

    def train_step(self, states, target_q_values, actions_one_hot, is_terminal_state):
        self._model.train_step((states, target_q_values, actions_one_hot, is_terminal_state))

    def copy_variables(self, other: 'Model'):
        self._model.set_weights(other._model.get_weights())

    def save(self, model_path):
        self._model.save(model_path)

    def save_weights(self, weights_path):
        self._model.save_weights(weights_path)

    def load_weights(self, weights_path):
        self._model.load_weights(weights_path)

    def _create_model(self):
        if DENSE_MODEL:
            inputs = layers.Input(shape=self._input_size)
            x = layers.Dense(128, activation='relu', kernel_initializer=VarianceScaling())(inputs)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation='relu', kernel_initializer=VarianceScaling())(x)
            outputs = layers.Dense(np.prod(self._output_size), activation='relu', kernel_initializer=VarianceScaling())(x)
            model = CustomKerasModel(inputs, outputs)

            model.compile(optimizer=optimizers.Adadelta(learning_rate=LEARNING_RATE),
                          loss=losses.Huber(),
                          metrics=['accuracy'])

        else:
            inputs = layers.Input(shape=self._input_size)
            x = layers.Conv2D(32, (3, 3), strides=(1, 1),
                              activation='relu',
                              kernel_initializer=GlorotUniform())(inputs)
            x = layers.Conv2D(64, (2, 2), strides=(1, 1),
                              activation='relu',
                              kernel_initializer=GlorotUniform())(x)
            x = layers.Flatten()(x)
            x = layers.Dense(256, activation='relu', kernel_initializer=GlorotUniform())(x)
            outputs = layers.Dense(np.prod(self._output_size), activation='relu', kernel_initializer=GlorotUniform())(x)
            model = CustomKerasModel(inputs, outputs)

            model.compile(optimizer=optimizers.RMSprop(
                            learning_rate=LEARNING_RATE, momentum=0.95
                          ),
                          loss=losses.Huber(),
                          metrics=['accuracy'])

        # model.summary()
        return model

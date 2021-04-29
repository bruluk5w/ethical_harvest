import numpy as np
from tensorflow import GradientTape, losses, squeeze, reduce_sum, convert_to_tensor
from tensorflow.keras import layers, optimizers
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Sequential

LEARNING_RATE = 0.001


class Model(Sequential):
    def __init__(self, input_size, output_size, name="SOAS_Model"):
        super().__init__(name=name)
        self._input_size = input_size
        self._output_size = output_size
        self.create_model()

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
            #todo: check, should be huber loss function
            target_q_values_t = convert_to_tensor(target_q_values)
            loss = self.compiled_loss(target_q_values_t, predicted_q_values, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(target_q_values_t, predicted_q_values)
        return {m.name: m.result() for m in self.metrics}

    def copy_variables(self, other: 'Model'):
        self.set_weights(other.get_weights())

    def create_model(self):
        self.add(layers.Conv2D(32, (3, 3), strides=(1, 1),
                               activation='relu',
                               # default parameters result in He initialization that work better with relu activation
                               kernel_initializer=VarianceScaling(),
                               input_shape=self._input_size))
        self.add(layers.Conv2D(64, (2, 2), strides=(1, 1),
                               activation='relu',
                               # default parameters result in He initialization that work better with relu activation
                               kernel_initializer=VarianceScaling()))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu', kernel_initializer=VarianceScaling()))
        self.add(layers.Dense(np.prod(self._output_size), activation='relu', kernel_initializer=VarianceScaling()))
        self.summary()

        self.compile(optimizer=optimizers.Adadelta(learning_rate=LEARNING_RATE),
                     loss=losses.Huber(),
                     metrics=['accuracy'])

    def get_config(self):
        return {'input_size': self._input_size, 'output_size': self._output_size}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

from keras.layers import SimpleRNN, Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from preprocesss import PreProcess
import numpy as np


class RNNClassifier(object):

    def __init__(self):
        self.time_steps = 10
        self.vector_length = 10
        pass

    def preprocess_data(self):

        preprocess = PreProcess('./data/us/covid/deaths.csv')
        data = preprocess.getData()
        death = data.iloc[:, 5:]
        FIPS = data['countyFIPS']

        X, label = [], []
        data = death.to_numpy()
        self.vector_length = len(data)
        for i in range(0, len(data[0]) - self.time_steps):

            x = data[:, i: i + self.time_steps]
            y = data[:, i + self.time_steps]
            X.append(x.T)
            label.append(y)
        
        return FIPS, np.asarray(X), np.asarray(label)

    def train(self):

        FIPS, X, y = self.preprocess_data()

        # X = np.random.random((100, self.time_steps, self.vector_length))
        # y = np.random.random((100, self.vector_length))

        input_data = Input(shape=(self.time_steps, self.vector_length,))

        rnn = SimpleRNN(
            units=128,
            activation='tanh'
        )(input_data)

        output = Dense(
            units=self.vector_length
        )(rnn)

        model = Model(input_data, output)

        optimizer = Adam(learning_rate=0.1)

        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        model.summary()

        callbacks = [
            ModelCheckpoint(
                'models/RNN/rnn_best.hdf5',
                monitor='loss',
                verbose=1,
                save_best_only=True,
                mode='auto',
                period=1
            ),
            EarlyStopping(
                monitor='loss',
                patience=5,
                mode='auto',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',
                patience=3,
                verbose=1,
                mode='auto',
                factor=0.1
            )
        ]

        model.fit(
            x=X,
            y=y,
            shuffle=True,
            batch_size=32,
            epochs=10,
            callbacks=callbacks,
            validation_split=0.2
        )

        model.save('models/RNN/rnn.hdf5')

        pass

    def test(self):

        FIPS, x, y = self.preprocess_data()

        X = np.reshape(x[-1], (1, self.time_steps, self.vector_length))

        model = load_model('models/RNN/rnn.hdf5')

        pre = model.predict(X)

        print(pre, len(pre[0]))


if __name__ == '__main__':
    clf = RNNClassifier()
    # clf.train()
    clf.test()

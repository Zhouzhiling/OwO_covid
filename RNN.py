from keras.layers import Input, Dense, LSTM
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from preprocesss import PreProcess
import numpy as np
import pandas as pd


class RNNClassifier(object):

    def __init__(self):
        self.time_steps = 2
        self.vector_length = 10
        self.pre_time_duration = 14
        pass

    def preprocess_data(self):

        preprocess = PreProcess('./data/us/covid/')
        data = preprocess.getNoneZeroData()
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

        rnn = LSTM(
            units=4 * self.vector_length,
            activation='tanh'
        )(input_data)

        # d = Dense(
        #     units=4 * self.vector_length,
        #     activation='relu'
        # )(rnn)

        d = Dense(
            units=2 * self.vector_length,
            activation='relu'
        )(rnn)

        output = Dense(
            units=self.vector_length
        )(d)

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
                patience=2,
                verbose=1,
                mode='auto',
                factor=0.1,
                min_lr=0.0001
            )
        ]

        model.fit(
            x=X,
            y=y,
            shuffle=True,
            batch_size=1,
            epochs=100,
            callbacks=callbacks
        )

        model.save('models/RNN/rnn.hdf5')

        pass

    def test(self):

        FIPS, X, y = self.preprocess_data()

        seed = np.reshape(X[-1], (1, self.time_steps, self.vector_length))

        model = load_model('models/RNN/rnn.hdf5')

        result = np.zeros((self.pre_time_duration, self.vector_length))

        for i in range(self.pre_time_duration):
            pre = model.predict(seed)

            result[i][:] = pre

            seed = np.reshape(np.concatenate((seed[0][1:][:], pre)), (1, self.time_steps, self.vector_length))

        prediction = pd.concat([FIPS, pd.DataFrame(result.T, dtype='int32')], axis=1)
        prediction.to_csv('processed_data/RNN/rnn_prediction.csv', index=False)


if __name__ == '__main__':
    clf = RNNClassifier()
    clf.train()
    clf.test()
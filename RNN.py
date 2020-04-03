from keras.layers import SimpleRNN, Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from preprocesss import PreProcess


class RNNClassifier(object):

    def __init__(self):
        self.time_steps = 10
        pass

    def preprocess_data(self):

        preprocess = PreProcess('./data/us/covid/deaths.csv')
        data = preprocess.getData()
        death = data.iloc[:, 4:]
        FIPS = data['countyFIPS']
        
        return death, FIPS

    def train(self):

        X, y = self.preprocess_data()

        input_data = Input(shape=(self.time_steps,))

        rnn = SimpleRNN(
            units=128,
            activation='tanh'
        )(input_data)

        output = Dense(
            units=1,
            activation='softmax'
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
                'rnn_best.hdf5',
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

        model.save('rnn.hdf5')

        pass

    def test(self):
        pass


if __name__ == '__main__':
    clf = RNNClassifier()
    clf.train()

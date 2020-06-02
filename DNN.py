from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from preprocessForNN import PreprocessForNN
import pandas as pd
import numpy as np


class DNN(object):

    def __init__(self, mode='outbreak'):
        self.preprocess = PreprocessForNN()
        self.mode = mode

    def train(self):

        feature, label = self.preprocess.generate_training_data(mode=self.mode)

        input_data = Input(shape=(53,))

        d = Dense(
            units=128,
            activation='relu'
        )(input_data)

        d = Dense(
            units=256,
            activation='relu'
        )(d)

        d = Dense(
            units=128,
            activation='relu'
        )(d)

        d = Dense(
            units=64,
            activation='relu'
        )(d)

        d = Dense(
            units=32,
            activation='relu'
        )(d)

        output = Dense(
            units=14
        )(d)

        model = Model(input_data, output)

        adam = Adam(learning_rate=0.0001)

        callbacks = [
            ModelCheckpoint(
                filepath='models/DNN/dnn.hdf5',
                save_best_only=True,
                verbose=1,
                period=1,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                verbose=1,
                factor=0.8,
                min_lr=0.0,
                mode='auto'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1
            )
        ]

        model.compile(
            optimizer=adam,
            loss='mse',
            metrics=['mse']
        )

        model.summary()

        model.fit(
            x=feature,
            y=label,
            epochs=100,
            verbose=1,
            shuffle=True,
            batch_size=4,
            callbacks=callbacks
        )

        model.save('models/DNN/dnn_model.hdf5')

    def predict(self):
        feature, FIPS, _ = self.preprocess.generate_testing_data(mode=self.mode)

        model = load_model('models/DNN/dnn_model.hdf5')
        pre = model.predict(feature)
        pre = np.round(pre * self.preprocess.get_std()[0] + self.preprocess.get_average()[0])

        prediction = pd.DataFrame(pre, index=None)

        result = pd.concat([FIPS, prediction], axis=1, ignore_index=True)
        result = result.rename(
            columns={
                0: 'countyFIPS',
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
                11: 10,
                12: 11,
                13: 12,
                14: 13
            }
        )

        path = 'models/DNN/dnn_' + self.mode + '.csv'
        result.to_csv(path, index=False)

        print('Predictions saved as ' + path)


if __name__ == '__main__':
    dnn = DNN()
    dnn.train()
    dnn.predict()

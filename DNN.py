from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from preprocessForNN import PreprocessForNN


class DNN(object):

    def __init__(self):
        self.preprocess = PreprocessForNN()

    def train(self):

        feature, label = self.preprocess.generate_training_data()

        input_data = Input(shape=(19,))

        d = Dense(
            units=32,
            activation='relu'
        )(input_data)

        d = Dense(
            units=16,
            activation='relu'
        )(d)

        output = Dense(
            units=14
        )(d)

        model = Model(input_data, output)

        adam = Adam(learning_rate=0.1)

        callbacks = [
            ModelCheckpoint(
                filepath='models/DNN/dnn.hdf5',
                save_best_only=True,
                verbose=1,
                period=5,
                monitor='loss'
            ),
            ReduceLROnPlateau(
                monitor='loss',
                patience=5,
                verbose=1,
                factor=0.8,
                min_lr=0.0001,
                mode='auto'
            ),
            EarlyStopping(
                monitor='loss',
                patience=10,
                verbose=1
            )
        ]

        model.compile(
            optimizer=adam,
            loss='mse',
            metrics='loss'
        )

        model.summary()

        model.fit(
            x=feature,
            y=label,
            epochs=100,
            verbose=1,
            validation_split=0.1,
            shuffle=True,
            batch_size=4,
            callbacks=callbacks
        )

        model.save('models/DNN/dnn_model.hdf5')

    def predict(self):
        feature, FIPS = self.preprocess.generate_testing_data()

        model = load_model('models/DNN/dnn_model.hdf5')
        pre = model.predct(feature)

        # todo save results

        return pre


if __name__ == '__main__':
    dnn = DNN()
    dnn.train()
    # dnn.predict()
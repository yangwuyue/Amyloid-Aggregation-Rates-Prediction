import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
from keras import regularizers

from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

SEED = 1530906270


class Data:

    def __init__(self, path):

        self.__path = path
        self.__test_indexes = None
        self.__train_x = []
        self.__train_y = []
        self.__test_x = []
        self.__test_y = []
        self.__data = []
        self.__label = []
        self.__read_file()

    def __read_file(self):
        with open(self.__path) as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines if line.strip() != '']
            data = []
            for line in lines:
                data.append([float(item) for item in line])
            self.__data = np.array(data)[:, :20]
            self.__label = np.array(data)[:, 20]
            self.__label = self.__label[:, np.newaxis]

    def __split_data(self):

        self.__train_x = []
        self.__train_y = []
        self.__test_x = []
        self.__test_y = []

        for index in range(len(self.__data)):
            if index not in self.__test_indexes:
                self.__train_x.append(self.__data[index].tolist())
                self.__train_y.append(self.__label[index].tolist())
            else:
                self.__test_x.append(self.__data[index].tolist())
                self.__test_y.append(self.__label[index].tolist())

    def get_data(self, test_indexes):

        self.__test_indexes = test_indexes
        self.__split_data()

        return self.__train_x, self.__train_y, self.__test_x, self.__test_y


class FCN:

    @staticmethod
    def my_loss(truth, predict):
        return K.mean(K.abs(truth-predict))

    @staticmethod
    def create_model(hide_node=30, regular_rate1=0, regular_rate2=0):
        inputs = Input(shape=(20,))
        x = Dense(hide_node, activation='relu', kernel_regularizer=regularizers.l2(regular_rate1))(inputs)
        outputs = Dense(1, kernel_regularizer=regularizers.l2(regular_rate2))(x)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=FCN.my_loss, metrics=["mean_absolute_error"])
        return model

    @staticmethod
    def load_model(model, weight_path):
        model.load_weights(weight_path)

    @staticmethod
    def main(train_data=None, train_label=None, val_data=None, val_label=None, key_name=None, seed=None, is_train=True, weight_path=None,
             param={"regular_rate1": 0,
                    "regular_rate2": 0,
                    "epoch": 1000,
                    "batch_size": 4,
                    "hide_node": 30}):
        global SEED
        if seed is not None:
            SEED = seed
        np.random.seed(SEED)
        param["seed"] = SEED
        param["keys_name"] = key_name
        model = FCN.create_model(param["hide_node"],
                                 param["regular_rate1"],
                                 param["regular_rate2"])
        callback = ModelCheckpoint('weight.hdf5', save_best_only=True, save_weights_only=True)
        if is_train:
            assert(train_data is not None and train_label is not None and val_data is not None and val_label is not None)
            model.fit(train_data, train_label,
                      batch_size=param["batch_size"],
                      validation_data=(val_data, val_label),
                      epochs=param["epoch"],
                      callbacks=[callback])
        else:
            FCN.load_model(model, weight_path)
        if not is_train:
            predict = model.predict(val_data)
            return predict
        return None


class MatchingLearningModel:

    @staticmethod
    def svr(kernel):
        return SVR(kernel=kernel)

    @staticmethod
    def line_regression():
        return LinearRegression()

    @staticmethod
    def random_forest(estimators=100, criterion="mse", max_features="auto"):
        return RandomForestRegressor(n_estimators=estimators,
                                     criterion=criterion,
                                     max_features=max_features)

    @staticmethod
    def lasso(alpha=0.1):
        return Lasso(alpha=alpha)

    @staticmethod
    def predict(model, train_data, train_label, test_data):
        model.fit(train_data, train_label)
        predict_value = model.predict(test_data)
        return predict_value

    @staticmethod
    def mse(truth, predict):

        mes_value = 0.0
        for index in range(len(truth)):
            mes_value += pow(truth[index] - predict[index], 2)
        return mes_value / len(truth)


def train_fcn(train_data, train_label, val_data, val_label):

    _ = FCN.main(train_data, train_label, val_data, val_label, is_train=True)
    K.clear_session()

if __name__ == "__main__":

    d = Data("Protein_kinetic_data.txt")
    ss_x = preprocessing.StandardScaler()
    ss_y = preprocessing.StandardScaler()
    test_index = range(140, 162) #Test Data
    train_x, train_y, val_x, val_y = d.get_data(test_index)
    train_x = ss_x.fit_transform(train_x)
    val_x = ss_x.transform(val_x)
    train_y = ss_y.fit_transform(train_y)
    val_y = ss_y.transform(val_y)
    train_fcn(train_x, train_y, val_x, val_y)

    fcn = FCN.main(val_data=val_x, is_train=False, weight_path="weight.hdf5")
    linear_svr = MatchingLearningModel.predict(MatchingLearningModel.svr("linear"), train_x, train_y, val_x)
    poly_svr = MatchingLearningModel.predict(MatchingLearningModel.svr("poly"), train_x, train_y, val_x)
    linear = MatchingLearningModel.predict(MatchingLearningModel.line_regression(), train_x, train_y, val_x)
    random_forest = MatchingLearningModel.predict(MatchingLearningModel.random_forest(), train_x, train_y, val_x)

    fcn = ss_y.inverse_transform(fcn)
    linear_svr = ss_y.inverse_transform(linear_svr)
    poly_svr = ss_y.inverse_transform(poly_svr)
    linear = ss_y.inverse_transform(linear)
    random_forest = ss_y.inverse_transform(random_forest)
    val_y = ss_y.inverse_transform(val_y)

    fcn_mse = MatchingLearningModel.mse(val_y, fcn)
    linear_svr_mse = MatchingLearningModel.mse(val_y, linear_svr)
    poly_svr_mse = MatchingLearningModel.mse(val_y, poly_svr)
    linear_mse = MatchingLearningModel.mse(val_y, linear)
    random_forest_mse = MatchingLearningModel.mse(val_y, random_forest)

    ###################*******MSE******###########################################
    print("fcn mse = %.3f\nlinear svr mse = %.3f\npoly svr mse = %.3f\nrandom forest mse = %.3f\nlinear mse = %.3f"
          % (fcn_mse, linear_svr_mse, poly_svr_mse, random_forest_mse, linear_mse))

from keras.models import Model, Sequential
from keras.layers import Dense


def NN() -> Model:
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=8))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def NN_Alzheimer() -> Model:
    model = Sequential()
    model.add(Dense(32,activation='relu',input_dim=9))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    return model
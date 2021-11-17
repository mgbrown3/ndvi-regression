from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

class Regressor(object):

    def __init__(self, alg, **kwargs):
        self.model = None
        if alg == "linear":
            self.model = LinearRegression(**kwargs)
        if alg == "randomforest":
            self.model = RandomForestRegressor(**kwargs)
    
    def train(self, xTrain, yTrain, modelFile=None):
        m = self.model.fit(xTrain, yTrain)

        if modelFile is not None:
            dump(m, modelFile)
        return m

    def _load_model(self, modelFile):
        return load(modelFile)


        


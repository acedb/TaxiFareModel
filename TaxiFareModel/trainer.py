# imports
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse

class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distance_pipe = make_pipeline(DistanceTransformer(), StandardScaler())

        time_pipe = make_pipeline(
            TimeFeaturesEncoder(time_column = 'pickup_datetime'),
            OneHotEncoder(handle_unknown = 'ignore')
            )

        preprocessor = ColumnTransformer([
            ('distance_trans', distance_pipe, ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']),
            ('time_trans', time_pipe, ['pickup_datetime'])])

        model_pipeline = Pipeline(steps = [('preprocessing', preprocessor),
                                            ('regressor', LinearRegression())])

        return model_pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get & clean data
    data = clean_data(get_data())

    # set X and y
    X = data.drop(columns = ['fare_amount'])
    y = data['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    trainer = Trainer(X_train, y_train)
    trainer.run()
    trainer.evaluate(X_test, y_test)
    # build pipeline
    #train_pipe = Trainer(X_train, y_train).set_pipeline()

    # train the pipeline
    #model = pipeline.run(X_train, y_train, train_pipe)

    # evaluate the pipeline
    #result = pipeline.evaluate(X_test, y_test, model)

    print(trainer.evaluate(X_test, y_test))

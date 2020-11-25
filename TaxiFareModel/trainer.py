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

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = 'https://mlflow.lewagon.co/'
myname = 'Alice'
EXPERIMENT_NAME = f'TaxifareModel_{myname}'

class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

        @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



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

        self.pipeline = model_pipeline

        return self

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(f'ID:{trainer.mlflow_experiment_id}')
        self.mlflow_log_param('model', str(self.pipeline.get_params()['model'])
                              .strip('()'))
        self.mlflow_log_metric('rmse', rmse)
        return rmse


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

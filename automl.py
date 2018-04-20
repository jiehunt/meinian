ddd=train.select_dtypes(include=['float64'])
a,b=train_test_split(ddd,test_size=0.2)
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
predictions=pd.DataFrame()
for classname in target:
    column_descriptions = {
        classname:'output',
    }

    ml_predictor = Predictor(type_of_estimator = 'regressor',column_descriptions = column_descriptions)
    ml_predictor.train(a,model_names=['LGBMRegressor'])
    from auto_ml.utils_models import load_ml_model
    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)
    predictions[classname] = trained_model.predict(b)

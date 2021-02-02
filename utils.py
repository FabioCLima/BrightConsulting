#! Bibliotecas que serão usadas para treinar o modelo 
import os
import pandas as pd 
import numpy as np
import sklearn 
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, max_error, \
explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


#! Funções utilizadas para criar as features e aplicar o pipeline de preprocessamento
#! 
# 1 - Separação das features por tipo - categóricas e numéricas

def split_type_features(df, target):
    """[Função que separa as features em categóricas e numéricas]

    Args:
        df ([pandas dataframe]): [DataFrame do pandas que tem as features e o target]
        target ([type]): [variável do tipo numérica, que é o variável dependente]

    Returns:
        [features]: [as features separadas por tipo - categóricas e numéricas]
    """    


    numeric_features = df.drop(target, axis = 1).\
        select_dtypes(include=['int64', 'float64']).columns

    categorical_features = df.select_dtypes(exclude=['number']).columns

    return numeric_features, categorical_features
#******************************************************************************

def apply_pipeline(num_features, cat_features):
    
# Preprocessamento
#
# Pipeline de processamento dos dados para gerar o modelo 
#

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])
    
    return preprocessor
#******************************************************************************

#! Dataframe da predição(obtido com o estimador)
def predict_dataframe(model_name, X_test, y_test):
    
    y_pred = model_name.predict(X_test)
    residuals = y_test - y_pred
    df_pred_actual = pd.DataFrame({
    'predicted' : np.round(y_pred, 3),   # preço estimado
    'actual' : np.round(y_test, 3),      # preço real
    'residuals': np.round(residuals, 3)  # diferença
    })
    return df_pred_actual.reset_index(drop=True)

#******************************************************************************

#! Métricas do modelo
#
def RMSE(y_test, y_pred):
    """
    Calculates Root Mean Squared Error between the actual and the predicted labels.
    
    """
    RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
    return RMSE


# Função para calcular as métricas do modelo
#
def show_scores(model, name, X, y, X_train, X_test, y_train, y_test):

    k = X.shape[1]
    n = len(y)
    y_pred = model.predict(X_test)
    

    #!explained variance score
    evars = explained_variance_score(y_test, y_pred)
    
    #! maximum residual error
    max_erro = max_error(y_test, y_pred)
    
    #! mean square error
    mse = mean_squared_error(y_test, y_pred)
    
    #! coeficiente de correlação
    r2_training = model.score(X_train, y_train)
    r2_true = r2_score(y_test, y_pred)
    
    #! r2 score ajustado para o número de features
    #! é a única métrica aqui que considera o problema de overfitting 
    r2_adjusted = 1 - ( (1 - r2_true) * (n - 1)/ (n - k - 1) )
    rmse = RMSE(y_test, y_pred)
    
    #! mean absolute percentual error
    mape = np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100
     
    scores = {
        "model name" : name, 
        "R2_score(training data)" : np.round(r2_training, 3),
        "R2_score(test data)" : np.round(r2_true, 3),
        "R2_adjusted" : np.round(r2_adjusted, 3),
        "Explained Variance Score" : np.round(evars, 3),
        "Maximum Residual Error" : np.round(max_erro, 3),
        "Mean Square Error" : np.round(mse, 3),
        "Root Mean Square Error" : np.round(rmse, 3),
        "Mean Absolute Percentual Error(%)" : np.round(mape, 3)
    }
    metricas = pd.Series(scores)
    return metricas
#******************************************************************************
# Outra maneira de avaliar utilizando crossvalidation
from sklearn.model_selection import cross_val_score
def display_scores(model,X_train, y_train):


    scores = cross_val_score(model, 
                         X_train, 
                         y_train, 
                         scoring = 'neg_mean_squared_error',
                         cv = 10)

    model_rmse_scores = np.sqrt(-1*scores)
    SCORES_MODEL = model_rmse_scores
    print("Scores: ", SCORES_MODEL)
    print("Mean: ", SCORES_MODEL.mean())
    print("Standard deviation:", SCORES_MODEL.std())
    
#**********************************************


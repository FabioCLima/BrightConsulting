from utils import*


#! Constantes de endereço
#
SRC_DIR = os.path.join( os.path.abspath('..'), 'src')
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMGS_DIR = os.path.join(BASE_DIR, 'imgs')
MODELS_DIR = os.path.join( BASE_DIR, 'models' )

input_file = "cars_to_modeling.csv"
input_data_modeling_path= os.path.join(DATA_DIR, input_file)
df = pd.read_csv(input_data_modeling_path, index_col=0)

num_features, cat_features = split_type_features(df, target='prices')

#! Separando em conjunto de treino e teste

X = df.drop('prices', axis= 1)
y = df['prices']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2021)
preprocessor = apply_pipeline(num_features, cat_features)

#! O estimador que será usado para treinar o modelo
#! já setado com os melhores hyperparâmetros
#*
xgb_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb_reg', xgb.XGBRegressor( booster = 'gbtree',
                                  objective = "reg:squarederror", 
                                  seed = 2021,
                                  colsample_bytree = 0.3, 
                                  gamma = 10,
                                  importance_type = 'gain', 
                                  learning_rate = 0.015,
                                  max_depth=10,
                                  min_child_weight=0.9,
                                  n_estimators=2000,
                                  reg_alpha=0, 
                                  reg_lambda=10, 
                                  subsample=0.9))
])

xgb_reg.fit(X_train, y_train)

df_pred_actual = predict_dataframe(xgb_reg, X_test, y_test)

print(df_pred_actual.head(10))
print("\nMétricas do modelo :")
algoritmo = "XGboost Regression tunned"
scores = show_scores(xgb_reg, algoritmo, X, y, X_train, X_test, y_train, y_test)
print(scores)
print("")
print(display_scores(xgb_reg, X_train, y_train))

#! Salving the model 
#
k = X.shape[1]
n = len(y)
y_pred = xgb_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2_true = r2_score(y_test, y_pred)
#  é a única métrica aqui que considera o problema de overfitting 
r2_adjusted = 1 - ( (1 - r2_true) * (n - 1)/ (n - k - 1) )
# mean absolute percentual error
mape = np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100

model_data = pd.Series({
    'features' : X.columns.tolist(),
    'model'    : xgb_reg,
    'score'    : r2_true,
    'r2_adjusted': r2_adjusted,
    'RMSE'     : rmse,
    'MAPE'     : mape,
})

model_data.to_pickle( os.path.join( MODELS_DIR, 'xgb_model.pkl' ) )
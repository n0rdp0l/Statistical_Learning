# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from pygam import LinearGAM, s, f 
import statsmodels.api as sm
import pylab as py
import xgboost as xgb
from xgboost import XGBRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import plot_importance
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
import keras_tuner
print(tf.__version__)
print(keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# %%
# open MHpredict csv file
MHpredict = pd.read_csv('MHpredict - Copy.csv')

# transform all categoricalvalues using LabelEncoder
## create a LabelEncoder object
le = LabelEncoder()

## iterate over the columns of the dataframe
## create empy list to store dtype of columns
type_cols = []
for col in MHpredict.columns:
  ## append the dtype of the column to the list
  type_cols.append(MHpredict[col].dtype)
  ## check if the column is of dtype object
  if MHpredict[col].dtype == 'object':
    ## fit and transform the column using the LabelEncoder
    MHpredict[col] = le.fit_transform(MHpredict[col])

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the DataFrame
scaled_df = scaler.fit_transform(MHpredict)

# Update the DataFrame with the scaled values
MHpredict_scaled = pd.DataFrame(scaled_df, columns=MHpredict.columns)

#  randomly split the dataset into a training (𝑛 = 1000) and test (𝑛 = 152)
train = MHpredict.sample(n=1000, random_state=3337731)
test = MHpredict.drop(train.index)

# splitting the training set into a training (80%) and validation set (20%)
#train, val = train_test_split(train, test_size=0.2, random_state=3337731)





# %%
MHpredict.describe()

# %%
#sns.pairplot(MHpredict)

# %%

y_train = train['dep_sev_fu']
X_train = train.drop('dep_sev_fu', axis=1)

y_test = test['dep_sev_fu']
X_test = test.drop('dep_sev_fu', axis=1)

#y_val = val['dep_sev_fu']
#X_val = val.drop('dep_sev_fu', axis=1)



# %%
### Generative Addative Model
# plot distribution of ordered dep_sev_fu values in MHpredict
# py.hist(MHpredict['dep_sev_fu'], bins=10, density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
#as well as qq plot:
#sm.qqplot(MHpredict['dep_sev_fu'], line ='45')


splines = 10 # aslo tried 5, 15, 20 and 30

gam = LinearGAM(f(0)+f(1)+s(2,n_splines=splines)+s(3,n_splines=splines)+s(4,n_splines=splines)+s(5,n_splines=splines)+
s(6,n_splines=splines)+s(7,n_splines=splines)+f(8)+f(9)+f(10)+f(11)+f(12)+f(13)+f(14)+s(15,n_splines=splines)+f(16)+f(17)+f(18)+f(19)).fit(X_train, y_train)
gam.gridsearch(X_train, y_train, progress=True,return_scores=True)
gam.summary()


# %%
titles = MHpredict.columns
# Set the number of rows and columns for the subplot
n_rows = 5
n_cols = 4

# Set the figure size
plt.figure()

# Create the subplot grid
fig, axs = plt.subplots(n_rows, n_cols,figsize=(30, 20))

# Loop through each subplot
for i, ax in enumerate(axs.flatten()):
    # Generate X values for the smooth function
    XX = gam.generate_X_grid(term=i)

    # Plot the partial dependence and 95% confidence interval
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

    # Set the title of the subplot
    ax.set_title(titles[i])


# %%
# performance of the model
y_pred = gam.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print the evaluation metrics
print("root mean squared error:", rmse)
print("mean absolute error:", mae)
print("R^2 score:", r2)

# %%
# devide rmse by range of dep_sev_fu to get a better understanding of the performance (comparing scalled and unscalled data)
print("root mean squared error divided by range of dep_sev_fu:", rmse/(max(y_test)-min(y_test)))

# %%
### boosting with XGBoost

#cross validation with XGBoost

# Set up the parameters to search over



hp_space = {'max_depth': [3, 4, 5, 6, 7, 8, 9], 
                'gamma': [0, 0.5, 1, 1.5, 2, 5], 
                'subsample': [0.6, 0.7, 0.8, 0.9, 1], 
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_bin' : [256, 512, 1024]
               }

xgb = XGBRegressor(seed = 3337731, tree_method = "hist")
boost = RandomizedSearchCV(estimator = xgb, 
                   param_distributions = hp_space,
                   n_iter = 10, 
                   scoring = 'neg_root_mean_squared_error',
                   cv = 10,
                   verbose=0)
boost.fit(X_train, y_train,
        
    )

print("Best parameters:", boost.best_params_)
print("Lowest RMSE: ", -boost.best_score_)


# %%
# split the X_train and y_train into a training (80%) and validation set (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=3337731)

xg_boost = XGBRegressor(
    subsample= 1, 
    max_depth=8,
    max_bin=256, 
    learning_rate= 0.1, 
    gamma= 2,
    colsample_bytree= 0.6,  
    seed=3337731)



xg_boost.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_val, y_val)], 
    verbose=True, 
    early_stopping_rounds = 20)

# %%
y_pred = boost.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print the evaluation metrics
print("root mean squared error:", rmse)
print("mean absolute error:", mae)
print("R^2 score:", r2)
fig, ax = plt.subplots(1,1,figsize=(10,14))
plot_importance(booster=xg_boost, ax=ax, importance_type='gain')

# %%
fig, ax = plt.subplots(1,1,figsize=(10,14))
plot_importance(booster=xg_boost, ax=ax,)

# %%
# devide rmse by range of dep_sev_fu
print("root mean squared error divided by range of dep_sev_fu:", rmse/(max(y_test)-min(y_test)))

# %%
### MLP with Keras (ANN)
# keras tuner is used to find the best hyperparameters for the model
# build the model
hp = keras_tuner.HyperParameters()
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=20))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", default=2, min_value=1, max_value=4)):
        if i==0:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+1}",default=128, min_value=30, max_value=515, step=25),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        if i==1:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+1}",default=256, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        if i==2:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+1}",default=256, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))

        
        else:
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i+1}",default=100, min_value=30, max_value=515, step=50),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                ))
        
        if hp.Boolean(f"dropout_{i+1}", default=True):
            model.add(layers.Dropout(rate=hp.Choice(f'rate_{i+1}', default=0.25, values = [0.25,0.5,0.75])))



    model.add(layers.Dense(1, activation="linear"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()],
    )
    return model

build_model(keras_tuner.HyperParameters())


# %%
# define the tuner
Tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=30,
    num_initial_points=2,
    seed=3337731,
    directory="ANN",
    project_name="MLP",
    #overwrite=True,
)

# %%
# convert train and test set to numpy arrays
X_train_np = X_train.to_numpy().astype('float32')
y_train_np = np.array(y_train).astype('float32')
X_test_np = np.array(X_test).astype('float32')
y_test_np = np.array(y_test).astype('float32')

#create validation set from the training set (20%)
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_train_np, y_train_np, test_size=0.2, random_state=3337731)

# %%
Tuner.search(X_train_np, y_train_np, epochs=200,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)],
            validation_data=(X_val_np, y_val_np),batch_size=20)

# %%
models_mlp = Tuner.get_best_models(num_models=3)
best_model_mlp = models_mlp[0]

best_model_mlp.summary()
Tuner.results_summary(1)
best_model_mlp.evaluate(X_test_np, y_test_np)

# %%
### Support Vector regression (this can be ignored)
# Create a nonlinear soft-margin SVR model
model = svm.SVC(kernel='rbf', gamma='auto')

# Define the hyperparameters to tune
hyperparameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

# Perform grid search using 5-fold cross validation
clf = GridSearchCV(model, hyperparameters, cv=10, verbose=0)

# Fit the model on the training and validation data
best_model = clf.fit(X_train, y_train)

# Print the best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Gamma:', best_model.best_estimator_.get_params()['gamma'])

# Print the accuracy score
print('Performance:', best_model.score(X_test, y_test))


# %%
# load DSthymia - Copy.csv
DSthymia = pd.read_csv('DSthymia - Copy.csv')
DSthymia

# %%
for col in DSthymia.columns:
  ## check if the column is of dtype object
  if DSthymia[col].dtype == 'object':
    ## fit and transform the column using the LabelEncoder
    DSthymia[col] = le.fit_transform(DSthymia[col])

# %%
y_gam = gam.predict(DSthymia)
y_boost = boost.predict(DSthymia)
y_mlp = best_model_mlp.predict(DSthymia.to_numpy().astype('float32'))

# %%
print(y_gam)
print(y_boost)
y_mlp



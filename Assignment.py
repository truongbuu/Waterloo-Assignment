
# coding: utf-8

# # _ ECE WATERLOO ASSIGNMENT _

# In[ ]:

# Import libaries
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = [15,18]
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


# _ Read modified csv file _

# In[ ]:

df = pd.read_csv('Data.csv')


# In[ ]:

#df = df[df.t != 0]


# In[ ]:

df.head()


# In[ ]:

#Split the input and output from dataframe
xy_output = df[['x_out','y_out']].values
df = df.drop(['x_out','y_out'],1)
X = df.values


# _ Features Transformation _

# In[ ]:

poly = PolynomialFeatures(6)
X =poly.fit_transform(X)


# In[ ]:

X[0]


# Split data for training and test 

# In[ ]:

X_train, X_test, xy_pos_train, xy_pos_test = train_test_split(X, xy_output, test_size = 0.2, random_state = 5 )


# # _ Apply Ridge Regression as our model_

# In[ ]:

Ridge_model_x = RidgeCV(alphas=(0.0001,0.001,0.01,0.1, 1.0, 10.0,100),cv = 5)
Ridge_model_x.fit(X_train,xy_pos_train[:,0])


# In[ ]:

Ridge_model_y = RidgeCV(alphas=(0.0001,0.001,0.01,0.1, 1.0, 10.0,100),cv = 5)
Ridge_model_y.fit(X_train,xy_pos_train[:,1])


# In[ ]:

print 'Coefficient for x with Ridge Regression:', Ridge_model_x.coef_
print 'Coefficient for y with Ridge Regression:', Ridge_model_y.coef_


# In[ ]:

i = Ridge_model_x.predict(X_train)
o = Ridge_model_x.predict(X_test)
print 'Root Mean Square Error for In-Sample of x with Ridge Regression :', np.sqrt(mean_squared_error(xy_pos_train[:,0], i))
print 'Root Mean Square Error for Out-Sample of x with Ridge Regression : ', np.sqrt(mean_squared_error(xy_pos_test[:,0], o))


# In[ ]:

i = Ridge_model_y.predict(X_train)
o = Ridge_model_y.predict(X_test)
print 'Root Mean Square Error for In-Sample of y with Ridge Regression ', np.sqrt(mean_squared_error(xy_pos_train[:,1], i))
print 'Root Mean Square Error for Out-Sample of y with Ridge Regression: ', np.sqrt(mean_squared_error(xy_pos_test[:,1], o))


# # _Apply Lasso Regression as our model_#

# In[ ]:

Lasso_model_x = LassoCV(alphas=[0.000001,0.00001, 0.0001,0.001,0.01,0.1, 1.0, 10.0,100],cv = 5)
Lasso_model_x.fit(X_train,xy_pos_train[:,0])


# In[ ]:

Lasso_model_y = LassoCV(alphas=[0.000001,0.00001, 0.0001,0.001,0.01,0.1, 1.0, 10.0,100],cv = 5)
Lasso_model_y.fit(X_train,xy_pos_train[:,1])


# In[ ]:

print 'Coefficient for x with Lasso Regression:', Lasso_model_x.coef_
print 'Coefficient for y with Lasso Regression:', Lasso_model_y.coef_


# In[ ]:

i = Lasso_model_x.predict(X_train)
o = Lasso_model_x.predict(X_test)
print 'Root Mean Square Error for In-Sample of x with Lasso Regression :', np.sqrt(mean_squared_error(xy_pos_train[:,0], i))
print 'Root Mean Square Error for Out-Sample of x with Lasso Regression : ', np.sqrt(mean_squared_error(xy_pos_test[:,0], o))


# In[ ]:

i = Lasso_model_y.predict(X_train)
o = Lasso_model_y.predict(X_test)
print 'Root Mean Square Error for In-Sample of y with Lasso Regression :', np.sqrt(mean_squared_error(xy_pos_train[:,1], i))
print 'Root Mean Square Error for Out-Sample of y with Lasso Regression : ', np.sqrt(mean_squared_error(xy_pos_test[:,1], o))


# # _Predict the trajectory of a projectile launched at 45 degrees with an initial velocity of 10 m/s till it hits the ground or timeindex=100 whichever is earlier_

# In[ ]:

#Read the formatted output file
predict_df = pd.read_csv('out_45_10.csv')
predict_df.head()


# In[ ]:

#Read the prepared ground truth
ground_truth = pd.read_csv('Ground_truth.csv')
X_truth = ground_truth['x'].values
Y_truth = ground_truth['y'].values
Y_truth_stop = Y_truth[Y_truth>=0]
X_truth_stop = X_truth[0:len(Y_truth_stop)]
ground_truth.head()


# In[ ]:

X_input = poly.fit_transform(predict_df.values)


# In[ ]:

x_output_Ridge = Ridge_model_x.predict(X_input)
y_output_Ridge = Ridge_model_y.predict(X_input)
y_output_Ridge_stop = y_output_Ridge[y_output_Ridge>=0]
x_output_Ridge_stop = x_output_Ridge[0:len(y_output_Ridge_stop)]
print x_output_Ridge_stop
print y_output_Ridge_stop


# In[ ]:

x_output_Lasso = Lasso_model_x.predict(X_input)
y_output_Lasso = Lasso_model_y.predict(X_input)
_less_than_0= np.argmax(y_output_Lasso < 0)
print _less_than_0
y_output_Lasso_stop = y_output_Lasso[0:_less_than_0]
x_output_Lasso_stop = x_output_Lasso[0:_less_than_0]
print x_output_Lasso
print y_output_Lasso





# # _Output csv file for submission (Ridge Regression only)_#

# In[ ]:

predict_df['x_out'] = x_output_Ridge
predict_df['y_out'] = y_output_Ridge
tem_df = pd.DataFrame([[0,0,0,0,0]], columns=predict_df.columns)


# In[ ]:

tem_df = tem_df.append(predict_df,ignore_index=True)
out_df = tem_df[tem_df['y_out'] >= 0]


# In[ ]:

out_df = out_df[['t','x_out','y_out']]
out_df.columns = ['[time_collect]', '[x]', '[y]']


# In[ ]:

out_df


# In[ ]:

out_df.to_csv('submission.csv', index= False)


# In[ ]:




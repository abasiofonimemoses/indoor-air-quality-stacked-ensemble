# Import necessary libraries

import pandas as pd # Used for data manipulation 

import seaborn as sns # Used for data visualization

import warnings # Used to handle warnings


# Ignore all warnings

warnings.filterwarnings("ignore") # Suppress warnings for cleaner output



# Load the dataset

indoor_air = pd.read_csv('/content/indoor_air.csv') # Read data from CSV file



# Display the first 10 rows of the dataset

indoor_air.head(10) # Show initial data for inspection



# Drop the 'ts' column

indoor_air = indoor_air.drop('ts', axis=1) # Remove the 'ts' column



# Check for missing values

pd.DataFrame(indoor_air.isnull().sum().sort_values(ascending=False)) # Display missing values per column
 
# Prepare features (X) and target (y)

X = indoor_air.drop(columns='pm25', axis=1) # Features for prediction



y = indoor_air['pm25'] # Target variable (PM2.5 levels)



# Split data into training and testing sets

from sklearn.model_selection import train_test_split # Import train_test_split function



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19) # Split data (80% train, 20% test)



# Linear Regression Model

from sklearn.svm import SVR # Import SVR



svr = SVR() # Create an SVR object



svr.fit(X_train, y_train)



y_pred_svr = svr.predict(X_test) # Make predictions on the test set



# Evaluate SVR Model

from sklearn.metrics import mean_squared_error, r2_score # Import evaluation metrics



mean_squared_error(y_test, y_pred_svr) # Calculate Mean Squared Error



r2_score(y_test, y_pred_svr) # Calculate R-squared
 
# LightGBM Model

import lightgbm as lgb # Import LightGBM


lgbm_reg = lgb.LGBMRegressor(random_state=19) # Create a LightGBMRegressor object



lgbm_reg.fit(X_train, y_train) # Train the LightGBM model



y_pred2 = lgbm_reg.predict(X_test) # Make predictions on the test set



# Evaluate LightGBM Model

mean_squared_error(y_test, y_pred2) # Calculate Mean Squared Error



r2_score(y_test, y_pred2) # Calculate R-squared



# Gradient Boosting Model

from sklearn.ensemble import GradientBoostingRegressor # Import GradientBoostingRegressor



gbr = GradientBoostingRegressor() # Create a Gradient Boosting Regressor object



gbr.fit(X_train, y_train) # Train the Gradient Boosting model



y_pred3 = gbr.predict(X_test) # Make predictions on the test set
 
# Evaluate Gradient Boosting Model

mean_squared_error(y_test, y_pred3) # Calculate Mean Squared Error



r2_score(y_test, y_pred3) # Calculate R-squared



from sklearn.ensemble import StackingRegressor # Import StackingRegressor 

from lightgbm import LGBMRegressor # Import LGBMRegressor for stacking 

from sklearn.linear_model import LinearRegression # Import LinearRegression


estimators = [('svr', SVR()), ('lgb', LGBMRegressor()), ('gbr', GradientBoostingRegressor())] # Define base estimators



sr = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())



sr.fit(X_train, y_train) # Train the Stacking Regressorestimators = [('svr', best_svr), ('lgb', best_lgb), ('gbr', best_gbr)]

sr	=	StackingRegressor(estimators=estimators, final_estimator=LinearRegression())	#	Assuming 'best_final_estimator' was meant to be LinearRegression

sr.fit(X_train, y_train)



y_pred4 = sr.predict(X_test) # Make predictions on the test set



# Evaluate Stacking Model

mean_squared_error(y_test, y_pred4) # Calculate Mean Squared Error 

r2_score(y_test, y_pred4) # Calculate R-squared


import matplotlib.pyplot as plt 

 
# Visualize predicted vs. actual values with distinction

plt.figure(figsize=(8, 6))



# Create scatter plot with green dots for predicted values

sns.scatterplot(x=y_test, y=y_pred4, color='green', label='Predicted') # Set color to green



# Add a straight line (diagonal) for reference

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Actual')



plt.xlabel("Actual pm25 Values") 
plt.ylabel("Predicted pm25 Values")
plt.title("Predicted vs. Actual pm25 Values (Stacked Model)")

plt.legend() # Show the legend (including the line label and predicted values label) plt.show()


from sklearn.ensemble import GradientBoostingRegressor 



# Assuming X_train, X_test, y_train, y_testare defined

gbr_reg = GradientBoostingRegressor(random_state=19).fit(X_train, y_train) # Train GBR 

y_pred_gbr = gbr_reg.predict(X_test) # Make predictions


plt.figure(figsize=(10, 6))

sns.scatterplot(x=y_test, y=y_pred_gbr, color='red', label='Predicted Values')
 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Ideal Prediction') 
plt.xlabel("Actual pm10 Levels")
plt.ylabel("Predicted pm10 Levels") 
plt.title("GBR Performance on pm10 Prediction") 
plt.legend()
plt.show()



# Make prediction for new input data

import numpy as np # Import NumPy for numerical operations 
input_data = (708, 72.09, 10.2, 20.83, 0.062) # New input data
input_data_as_numpy_array = np.asarray(input_data) # Convert to NumPy array 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) # Reshape for prediction 
prediction = sr.predict(input_data_reshaped) # Make prediction
print(prediction) # Print the prediction



# Categorize prediction based on PM2.5 levels

if prediction[0] >= 0 and prediction[0] <= 12.0:

    print("Good") # Good air quality

elif prediction[0] > 12.0 and prediction[0] <= 35.4:

    print("Moderate") # Moderate air quality

elif prediction[0] > 35.4 and prediction[0] <= 150.4:

    print("Bad") # Bad air quality
elif prediction[0] > 150.4 and prediction[0] <= 500.4: 
    print("Hazardous") # Hazardous air quality
else:

    print("The predicted PM2.5 value is outside the defined categories.") # Out of range

 
#SAVING THE TRAINED MODEL

# Save the trained model

import joblib # Import pickle for model serialization



filename = 'indoor_air_model.sav' # Filename for the saved model 
joblib.dump(sr, open(filename, 'wb')) # Save the model to a file


# Load the saved model

loaded_model = joblib.load(open('indoor_air_model.sav', 'rb')) # Load the model from file



# Make prediction using the loaded model

input_data = (625, 41.81, 19, 19.43, 0.062) # New input data input_data_as_numpy_array = np.asarray(input_data) # Convert to NumPy array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) # Reshape for prediction 
prediction = loaded_model.predict(input_data_reshaped) # Make prediction
print(prediction) # Print the prediction



# Categorize prediction based on PM2.5 levels

if prediction[0] >= 0 and prediction[0] <= 12.0: 
    print("Good") # Good air quality
elif prediction[0] > 12.0 and prediction[0] <= 35.4: 
    print("Moderate") # Moderate air quality
elif prediction[0] > 35.4 and prediction[0] <= 150.4: 
    print("Bad") # Bad air quality
 
elif prediction[0] > 150.4 and prediction[0] <= 500.4: print("Hazardous") # Hazardous air quality
else:

    print("The predicted PM2.5 value is outside the defined categories.") # Out of range

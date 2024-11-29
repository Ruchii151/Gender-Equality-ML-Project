

# In[35]:


#Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


#Loading and exploring the data 

data = pd.read_csv('gender_pay_gap_dataset.csv') 

# Display basic information about the data
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())


# In[37]:


# Exploratory Data Analysis (EDA)

# Check gender distribution
sns.countplot(x='gender', data=data)
plt.title('Gender Distribution')
plt.show()

# Visualize salary distribution by gender
sns.boxplot(x='gender', y='salary', data=data)
plt.title('Salary Distribution by Gender')
plt.show()

# Visualize pay gaps by industry
sns.boxplot(x='industry', y='salary', hue='gender', data=data)
plt.xticks(rotation=90)
plt.title('Pay Gaps by Industry')
plt.show()


# In[38]:


#Data Preprocessing

# Encode categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['industry'] = label_encoder.fit_transform(data['industry'])
data['job_role'] = label_encoder.fit_transform(data['job_role'])
data['region'] = label_encoder.fit_transform(data['region'])
data['education_level'] = label_encoder.fit_transform(data['education_level'])

# Feature scaling
scaler = StandardScaler()
data[['years_of_experience', 'salary']] = scaler.fit_transform(data[['years_of_experience', 'salary']])

# Split the data into features and target variable
X = data.drop('salary', axis=1)
y = data['salary']


# In[39]:


#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


#Model Training

# Define the model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred = xgb_model.predict(X_test)


# In[41]:


#Model evaluation

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")


# In[42]:


# Feature importance for interpretability
xgb.plot_importance(xgb_model)
plt.title('Feature Importance')
plt.show()


# In[ ]:





# In[ ]:





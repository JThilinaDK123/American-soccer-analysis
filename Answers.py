#!/usr/bin/env python
# coding: utf-8

# ### Load the Libraries

# In[1]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV


# ### Load the datasets

# In[2]:


# Path to the folder containing CSV files
folder_path = 'raw_soccer_files'

# List all files in the folder
file_list = os.listdir(folder_path)

# Create separate dataframes and name them with their original names
for file_name in file_list:
    if file_name.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file into a dataframe
        df = pd.read_csv(file_path)
        
        # Use the original file name (without extension) as the dataframe name
        df_name = os.path.splitext(file_name)[0]
        
        # Create a variable with the dataframe name and store the dataframe in it
        locals()[df_name] = df


# ### Info for each datasets

# #### 01) Player salary dataset

# In[3]:


player_salary_data


# In[4]:


# Drop 'mlspa_release' column from the dataset. 
# It creates same data row in multiple times. Therefore, its worth to remove that column from the dataset
player_salary_data = player_salary_data.drop(columns = ['mlspa_release','competition','guaranteed_compensation'])


# In[5]:


# Drop all the duplicates
player_salary_data = player_salary_data.drop_duplicates()
player_salary_data.shape


# In[6]:


player_salary_data.dtypes


# In[7]:


# Check for the missing values
player_salary_data.isna().sum()


# In[8]:


# Drop all the missing values
player_salary_data = player_salary_data.dropna()
player_salary_data.isna().sum()


# In[9]:


player_salary_data.shape


# #### 02) Game Info dataset

# In[10]:


game_info


# In[11]:


game_info.dtypes


# In[12]:


game_info.isna().sum() # No missing values


# In[13]:


game_info.duplicated().sum() # No duplicates


# #### 03) Player Info dataset

# In[14]:


player_info


# In[15]:


player_info.dtypes


# In[16]:


# Check for the missing values
player_info.isna().sum()


# In[17]:


# Drop all the missing values
player_info = player_info.dropna()
player_info.isna().sum()


# In[18]:


player_info.duplicated().sum() # No duplicates


# In[19]:


player_info['height'] = (player_info['height_ft'] * 12) + player_info['height_in']
player_info = player_info.drop(columns = ['height_ft','height_in'])
player_info


# In[20]:


player_info.shape


# #### 04) Player passing data

# In[21]:


player_passing_data


# In[22]:


player_passing_data = player_passing_data.drop(columns = ['competition'])


# In[23]:


player_passing_data.dtypes


# In[24]:


player_passing_data.isna().sum() # No missing values


# In[25]:


player_passing_data['minutes_played'] = player_passing_data['minutes_played'].astype(float)

# List of columns to be transformed
quantitative_columns = [
    'attempted_passes',
    'pass_completion_percentage',
    'xpass_completion_percentage',
    'passes_completed_over_expected',
    'passes_completed_over_expected_p100',
    'avg_distance_yds',
    'avg_vertical_distance_yds',
    'share_team_touches'
]

# Iterate through each row and update the values in the quantitative columns
for col in quantitative_columns:
    player_passing_data[col] = (player_passing_data[col] / player_passing_data['minutes_played']) * 90

player_passing_data


# In[26]:


player_passing_data.duplicated().sum() # No duplicates


# #### 05) Team Info dataset

# In[27]:


team_info


# In[28]:


team_info.dtypes


# In[29]:


team_info.isna().sum() # No missing values


# In[30]:


team_info.duplicated().sum() # No duplicates


# #### 06) player ga avg dataset

# In[31]:


player_ga_avg


# In[32]:


player_ga_avg.dtypes


# In[33]:


player_ga_avg.isna().sum() # No missing values


# In[34]:


player_ga_avg['minutes_played'] = player_ga_avg['minutes_played'].astype(float)

# List of columns to be transformed
quantitative_columns = [
 'goals_added_above_avg_Dribbling',
 'goals_added_above_avg_Fouling',
 'goals_added_above_avg_Interrupting',
 'goals_added_above_avg_Passing',
 'goals_added_above_avg_Receiving',
 'goals_added_above_avg_Shooting',
 'count_actions_Dribbling',
 'count_actions_Fouling',
 'count_actions_Interrupting',
 'count_actions_Passing',
 'count_actions_Receiving',
 'count_actions_Shooting',
 'offensive_goals_added']

# Iterate through each row and update the values in the quantitative columns
for col in quantitative_columns:
    player_ga_avg[col] = (player_ga_avg[col] / player_ga_avg['minutes_played']) * 90

player_ga_avg


# In[35]:


player_ga_avg.duplicated().sum() # No duplicates


# ### Create a single dataframe

# In[36]:


merged_df_1 = pd.merge(player_salary_data, player_info, on='player_id', how='inner')
merged_df_1


# In[37]:


# sample = merged_df_1[merged_df_1['player_id']=='Oa5wY8RXQ1']
# sample


# In[38]:


merged_df_2 = pd.merge(merged_df_1, team_info, on='team_id', how='inner')
merged_df_2


# In[39]:


# sample = merged_df_2[merged_df_2['player_id']=='Oa5wY8RXQ1']
# sample


# In[40]:


merged_df_3 = pd.merge(merged_df_2, player_ga_avg, on=['player_id', 'team_id', 'season_name'], how='inner')
merged_df_3


# In[41]:


# sample = merged_df_3[merged_df_3['player_id']=='Oa5wY8RXQ1']
# sample


# In[42]:


merged_df_4 = pd.merge(player_passing_data, game_info, on='game_id', how='inner')
merged_df_4


# In[43]:


# sample = merged_df_4[merged_df_4['player_id']=='Oa5wY8RXQ1']
# sample


# In[44]:


# Define custom aggregation functions
def custom_agg(x):
    if x.dtype == 'O':
        return x.mode().iloc[0]  # Mode for qualitative variables
    else:
        return x.sum()  # Sum for other quantitative variables

# Apply custom aggregation using the defined function
merged_df_4 = merged_df_4.groupby(['player_id', 'season_name'], as_index=False).agg(custom_agg)
merged_df_4


# In[45]:


# sample = merged_df_4[merged_df_4['player_id']=='Oa5wY8RXQ1']
# sample


# In[46]:


merged_df_5 = pd.merge(merged_df_4, merged_df_3, on=['player_id','team_id', 'season_name',
                                                      'general_position'], how='inner')
merged_df_5


# In[47]:


# Check for the missing values
merged_df_5.isna().sum()


# In[48]:


# Drop all the missing values
merged_df_5 = merged_df_5.dropna()
merged_df_5.isna().sum()


# In[49]:


merged_df_5.shape


# In[50]:


merged_df_5.duplicated().sum() # No duplicates


# ### Final Dataframe

# In[51]:


# merged_df_5['weight_lb'] = merged_df_5['weight_lb'].astype('object')
# merged_df_5['height'] = merged_df_5['height'].astype('object')


# In[52]:


# # Define custom aggregation functions
# def custom_agg(x):
#     if x.dtype == 'O':
#         return x.mode().iloc[0]  # Mode for qualitative variables
#     elif x.name in ['pass_completion_percentage', 'xpass_completion_percentage']:
#         return x.mean()  # Average for variables 'pass_completion_percentage' and 'xpass_completion_percentage'
#     else:
#         return x.sum()  # Sum for other quantitative variables

# # Apply custom aggregation using the defined function
# final_df = merged_df_5.groupby(['player_id', 'season_name'], as_index=False).agg(custom_agg)
# final_df


# In[53]:


# final_df['weight_lb'] = final_df['weight_lb'].astype('float')
# final_df['height'] = final_df['height'].astype('float')


# In[54]:


final_df = merged_df_5
final_df


# ## Visualization

# In[55]:


# Set the style for seaborn
sns.set(style="whitegrid")

# Plot a histogram for the target variable
plt.figure(figsize=(12, 6))
sns.histplot(data=final_df, x='offensive_goals_added', bins=30, kde=True)
plt.title('Distribution of the Target Variable')
plt.xlabel('offensive goals added')
plt.ylabel('Frequency')
plt.show()


# In[56]:


# Plot a histogram for minutes_played_x
plt.figure(figsize=(12, 6))
sns.histplot(data=final_df, x='minutes_played_x', bins=30, kde=True)
plt.title('Distribution of Minutes Played')
plt.xlabel('Minutes Played')
plt.ylabel('Frequency')
plt.show()


# In[57]:


# Select the first 10 nationalities
top_10_nationalities = final_df['nationality'].value_counts().index[:10]

# Filter the DataFrame to include only the top 10 nationalities
filtered_df = final_df[final_df['nationality'].isin(top_10_nationalities)]

# Plot the bar plot
plt.figure(figsize=(15, 8))
sns.countplot(data=filtered_df, y='nationality', order=top_10_nationalities, palette='viridis')
plt.title('Top 10 Nationalities Distribution')
plt.xlabel('Count')
plt.ylabel('Nationality')
plt.show()


# In[58]:


# Plot a line chart for pass_completion_percentage over seasons
plt.figure(figsize=(12, 6))
sns.lineplot(data=final_df, x='season_name', y='pass_completion_percentage')
plt.title('Pass Completion Percentage Over Seasons')
plt.xlabel('Season')
plt.ylabel('Pass Completion Percentage')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[59]:


# Boxplot for position vs pass_completion_percentage
plt.figure(figsize=(12, 6))
sns.boxplot(data=final_df, x='position', y='pass_completion_percentage')
plt.title('Pass Completion Percentage by Position')
plt.xlabel('Position')
plt.ylabel('Pass Completion Percentage')
plt.show()


# In[60]:


# Create a pairplot for selected numeric columns
numeric_cols = ['minutes_played_x', 'attempted_passes', 'avg_distance_yds', 'goals_added_above_avg_Passing']
sns.pairplot(final_df[numeric_cols])
plt.suptitle('Pairplot of Numeric Columns', y=1.02)
plt.show()


# In[61]:


# Line chart for share_team_touches over seasons
plt.figure(figsize=(12, 6))
sns.lineplot(data=final_df, x='season_name', y='share_team_touches')
plt.title('Share of Team Touches Over Seasons')
plt.xlabel('Season')
plt.ylabel('Share of Team Touches')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[62]:


# Scatter plot for minutes played vs goals_added_above_avg_Shooting
plt.figure(figsize=(12, 6))
sns.scatterplot(data=final_df, x='minutes_played_x', y='goals_added_above_avg_Shooting', hue='position', palette='Set2')
plt.title('Minutes Played vs Goals Added Above Average (Shooting)')
plt.xlabel('Minutes Played')
plt.ylabel('Goals Added Above Average (Shooting)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[63]:


# Select specific variables for density plots
selected_variables = ["goals_added_above_avg_Dribbling", "goals_added_above_avg_Fouling", 
                      "goals_added_above_avg_Interrupting", "goals_added_above_avg_Passing",
                      "goals_added_above_avg_Receiving", "goals_added_above_avg_Shooting"]

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

# Create density plots for selected variables using seaborn
for i, column in enumerate(selected_variables):
    sns.kdeplot(data=final_df, x=column, ax=axes[i], fill=True)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'Density Plot for {column}')

# Adjust layout
plt.tight_layout()
plt.show()


# In[64]:


# Select specific variables for density plots
selected_variables = ["count_actions_Dribbling", "count_actions_Fouling", 
                      "count_actions_Interrupting", "count_actions_Passing",
                      "count_actions_Receiving", "count_actions_Shooting"]

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

# Create density plots for selected variables using seaborn
for i, column in enumerate(selected_variables):
    sns.kdeplot(data=final_df, x=column, ax=axes[i], fill=True)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'Density Plot for {column}')

# Adjust layout
plt.tight_layout()
plt.show()


# ## Modelling

# In[65]:


# Take a copy of the "final_df"
final_df_sample = final_df.copy()


# In[66]:


# Variable Conversion
final_df_sample['season_name'] = final_df_sample['season_name'].astype('object')


# In[67]:


# Rename column names
final_df_sample = final_df_sample.rename(columns = {'minutes_played_x':'minutes_played'})


# In[68]:


final_df_sample.columns


# In[69]:


# Drop Unwanted columns
final_df_sample = final_df_sample.drop(columns = ['player_id' , 'game_id' , 'team_id' ,
                                    'team_short_name' , 'team_abbreviation', 'minutes_played_y' , 'date',
                                   'player_name' , 'birth_date' , 'team_name'])
final_df_sample


# In[70]:


# Label encoding for categorical variables
label_encoder = LabelEncoder()

categorical_columns = ['season_name','general_position','position','nationality'] 
for col in categorical_columns:
    final_df_sample[col] = label_encoder.fit_transform(final_df_sample[col])


# In[71]:


final_df_sample.dtypes


# In[72]:


final_df_sample


# In[73]:


# Filter data for training and testing
# 9 = 2022 (2022 becomes 9 because we did label encoding for the "session_name" variable)
train_data = final_df_sample[final_df_sample['season_name'] != 9]
test_data = final_df_sample[final_df_sample['season_name'] == 9]


# In[74]:


train_data


# In[75]:


test_data


# In[76]:


# Define features (X) and target variable (y)
X_train = train_data.drop(['offensive_goals_added'], axis=1)
y_train = train_data['offensive_goals_added']

X_test = test_data.drop(['offensive_goals_added'], axis=1)
y_test = test_data['offensive_goals_added']


# #### Check multicolinearity

# In[77]:


# Select only float64 variables as quantitative columns
num = ['float64']
num_vars = list(train_data.select_dtypes(include = num))


# In[78]:


num_vars = num_vars[0:24]
num_vars


# In[79]:


# Correlation matrix
corrmatrix_before = X_train[num_vars].corr()
corrmatrix_before


# In[80]:


# Set up the matplotlib figure
plt.figure(figsize=(18, 10))  # Adjust the figure size as needed

# Create a heatmap with seaborn
sns.heatmap(corrmatrix_before, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Set font size for annotations
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Show the plot
plt.show()


# In[81]:


# its hard for us to look into the correlation matrix
# By using below function, we can identify the variables that experiancing multicolinearity
def correlation(df, threshold):
    correlated_cols = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_cols.add(colname)
    return correlated_cols


# In[82]:


# following are the variables that experiancing multicolineairty issue (more than 0.5)
corre_feature = correlation(X_train[num_vars], 0.5)
corre_feature


# In[83]:


# Drop all the variables that has multicolinearity
X_train.drop(labels = corre_feature, axis = 1, inplace = True)
X_test.drop(labels = corre_feature, axis = 1, inplace = True)


# #### after removing multicolineairty

# In[84]:


X_train.columns


# In[85]:


num_vars_new = ['minutes_played',
       'pass_completion_percentage', 'passes_completed_over_expected',
       'passes_completed_over_expected_p100', 'avg_vertical_distance_yds',
       'base_salary','weight_lb',
       'goals_added_above_avg_Dribbling', 'goals_added_above_avg_Fouling',
       'goals_added_above_avg_Interrupting', 'goals_added_above_avg_Passing',
       'goals_added_above_avg_Receiving', 'count_actions_Dribbling',
       'count_actions_Fouling', 'count_actions_Interrupting',
       'count_actions_Passing']


# In[86]:


# Correlation matrix
corrmatrix_after = X_train[num_vars_new].corr()
corrmatrix_after


# In[87]:


# Set up the matplotlib figure
plt.figure(figsize=(18, 10))  # Adjust the figure size as needed

# Create a heatmap with seaborn
sns.heatmap(corrmatrix_after, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Set font size for annotations
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Show the plot
plt.show()


# In[88]:


X_train.shape


# In[89]:


X_test.shape


# ### 01) Random Forest Regressor

# In[90]:


# Random and Grid Search models
random_search_model = RandomForestRegressor(random_state=42)
grid_search_model = RandomForestRegressor(random_state=42)


# In[91]:


# Cross Valaidation
cv_split = KFold(n_splits=5, random_state=42, shuffle=True)


# In[92]:


# Hyper parameter Grid
rf_hyperparam_grid={
    'n_estimators': [25, 50],
    'bootstrap': [True]
}


# In[93]:


rf_random_search=RandomizedSearchCV(
    estimator=random_search_model,
    param_distributions=rf_hyperparam_grid,
    scoring="neg_log_loss",
    refit=True,
    return_train_score=True,
    cv=cv_split,    
    verbose=10,
    n_jobs=-1,
    random_state=42
)


# In[94]:


# Tuned Model
tuned_random_model_rf = rf_random_search.fit(X_train, y_train)


# In[95]:


# Best Parameters
tuned_random_model_rf.best_params_


# In[96]:


# Random Forest model with Best Parameters
model_rf = RandomForestRegressor(random_state=42, n_estimators = 25, bootstrap = True)
model_rf.fit(X_train, y_train)


# In[97]:


# Make predictions on the test set
predictions_rf = model_rf.predict(X_test)


# In[98]:


# Evaluate the model (for example, using mean squared error)
mse = mean_squared_error(y_test, predictions_rf)
print(f'Mean Squared Error on the test set: {mse}')

# Evaluate the model using R-squared
r2 = r2_score(y_test, predictions_rf)
print(f'R-squared on the test set: {r2}')


# In[99]:


# Visualize the predictions vs. actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions_rf)
plt.title('Predictions vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[100]:


# Get feature importances from the trained model
feature_importances = model_rf.feature_importances_

# Get the names of features
feature_names = X_train.columns

# Sort feature importances in descending order
indices = feature_importances.argsort()[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Variable Importance Plot")
plt.show()


# In[101]:


# Create a DataFrame with actual and predicted values
prediction_df_rf = pd.DataFrame({'offensive_goals_added': y_test, 'Predicted_for_2022': predictions_rf})
prediction_df_rf


# In[102]:


subset_df_2022 = final_df[final_df['season_name'] == 2022]
subset_df_2022


# In[103]:


# Final predictive dataset (here, we have only the players that play for the year 2022)
predictive_df_rf = pd.merge(subset_df_2022,prediction_df_rf, on = "offensive_goals_added" , how = "inner" )
predictive_df_rf = predictive_df_rf[["player_id", "player_name", "season_name", 
                              "offensive_goals_added", "Predicted_for_2022"]]
predictive_df_rf


# ### 02) XGBoost Regressor

# In[104]:


# Random and Grid Search models
random_search_model = XGBRegressor(random_state=42)
grid_search_model = XGBRegressor(random_state=42)


# In[105]:


# Hyper parameter Grid
xgb_hyperparam_grid={
    'depth': [3, 6],
    'subsample': [0.5, 0.6]
}


# In[106]:


xgb_random_search=RandomizedSearchCV(
    estimator=random_search_model,
    param_distributions=xgb_hyperparam_grid,
    scoring="neg_log_loss",
    refit=True,
    return_train_score=True,
    cv=cv_split,    
    verbose=10,
    n_jobs=-1,
    random_state=42
)


# In[107]:


# Tuned Model
tuned_random_model_xgb = xgb_random_search.fit(X_train, y_train)


# In[108]:


# Best Parameters
tuned_random_model_xgb.best_params_


# In[109]:


# Initialize and train the XGBoost model
model_xgb = XGBRegressor(random_state=42, depth = 3, subsample = 0.5)
model_xgb.fit(X_train, y_train)

# Make predictions on the test set
predictions_xgb = model_xgb.predict(X_test)

# Evaluate the XGBoost model (for example, using mean squared error)
mse_xgb = mean_squared_error(y_test, predictions_xgb)
print(f'Mean Squared Error on the test set (XGBoost): {mse_xgb}')

# Evaluate the model using R-squared
r2 = r2_score(y_test, predictions_xgb)
print(f'R-squared on the test set: {r2}')


# In[110]:


# Visualize the predictions vs. actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions_xgb)
plt.title('Predictions vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[111]:


# Plot Variable Importance
plot_importance(model_xgb)
plt.show()


# In[112]:


# Create a DataFrame with actual and predicted values
prediction_df_xgb = pd.DataFrame({'offensive_goals_added': y_test, 'Predicted_for_2022': predictions_xgb})
prediction_df_xgb


# In[113]:


# Final predictive dataset (here, we have only the players that play for the year 2022)
predictive_df_xgb = pd.merge(subset_df_2022,prediction_df_xgb, on = "offensive_goals_added" , how = "inner" )
predictive_df_xgb = predictive_df_xgb[["player_id", "player_name", "season_name", 
                              "offensive_goals_added", "Predicted_for_2022"]]
predictive_df_xgb


# In[114]:


sorted_df = predictive_df_xgb.sort_values(by='Predicted_for_2022', ascending=False)
sorted_df


# In[ ]:





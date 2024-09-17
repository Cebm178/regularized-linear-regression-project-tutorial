# Import Libraries
import pandas as pd
# We will commonly use pandas to handle dataframes, which is a crucial library for handling datasets in ML tasks.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Lasso
from pickle import dump

# import dataset(s)
df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv", sep = ",")
df.head()

df.info()

df.columns

# Explore dataset(s)
df = df.drop_duplicates().reset_index(drop = True)
df.head()
# Use .head(), .info(), and .describe() to get an initial understanding of your data.

# Preliminary Analysis
# Identify numeric columns, excluding 'Heart disease_number'
numeric_columns = df.select_dtypes(exclude="object").columns.drop("Heart disease_number")

# Scale the numeric columns using StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), 
                                 index=df.index, 
                                 columns=numeric_columns)

# Add the 'Heart disease_number' column back to the scaled DataFrame
df_scaled["Heart disease_number"] = df["Heart disease_number"]

# Display the first few rows of the scaled data
df.head()

# Define features (X) and target (y)
X = df_scaled.drop(columns=["Heart disease_number"])
y = df_scaled["Heart disease_number"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store training and testing indices (optional, kept for potential future use)
train_indices = X_train.index.tolist()
test_indices = X_test.index.tolist()

# Select top 30% of features using SelectKBest with f_regression
k = int(X_train.shape[1] * 0.3)  # Selecting 30% of features
selection_model = SelectKBest(score_func=f_regression, k=k)

# Fit the selector on training data and transform both train and test sets
X_train_sel = pd.DataFrame(selection_model.fit_transform(X_train, y_train), 
                           columns=X_train.columns[selection_model.get_support()])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), 
                          columns=X_test.columns[selection_model.get_support()])

# Show the first few rows of the selected features in the training set
X_train_sel.head()
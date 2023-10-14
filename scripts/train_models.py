import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , accuracy_score , confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('data/Placement_Data.csv')

#median_value = df['salary'].median()
df['salary'].fillna(0 , inplace = True)

df.to_csv('data/placement_data_cleaned.csv', index=False)

le = LabelEncoder()

for i in df.columns :
    if df[i].dtype == 'object' :
        df[i] = le.fit_transform(df[i])

X_salary = df[['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p','degree_t','specialisation']]
Y_salary = df['salary']

X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(X_salary, Y_salary, test_size=0.20, random_state=42)

# Train the salary prediction model
salary_model = HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100,tol=1e-05, warm_start=False)  # 'epsilon' is a parameter controlling the sensitivity to outliers

salary_model.fit(X_train_salary, y_train_salary)
salary_rmse = mean_squared_error(y_test_salary, salary_model.predict(X_test_salary), squared=False)


X_placed = df[['ssc_p','hsc_p','degree_p','workex','etest_p','mba_p','degree_t','specialisation']]
Y_placed = df['status']

X_train_placed, X_test_placed, y_train_placed, y_test_placed = train_test_split(X_placed, Y_placed, test_size=0.2, random_state=42)

# Train the placement prediction model
placement_model = GaussianNB(priors=None, var_smoothing=1e-09)
placement_model.fit(X_train_placed, y_train_placed)
placement_accuracy = accuracy_score(y_test_placed, placement_model.predict(X_test_placed))

# Save the models
joblib.dump(placement_model, 'models/placement_model.pkl')
joblib.dump(salary_model, 'models/salary_model.pkl')


print(f'Placement Model Accuracy: {placement_accuracy:.2f}')
print(f'Salary Model RMSE: {salary_rmse:.2f}')

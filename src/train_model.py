import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# load data
print("loading data")
if not os.path.exists('./data/processed/sample_df.csv'):
    print("need to run gatherData.py first")
    exit()
    
df = pd.read_csv('./data/processed/sample_df.csv')
print(f"loaded {len(df)} rows")

print(f"total accidents: {len(df)}")
print("severity distribution:")
print(df['Severity'].value_counts().sort_index())

# pick features (just starting with weather)
features = ['Temperature(F)', 'Visibility(mi)', 'Distance(mi)', 'Humidity(%)', 'Pressure(in)']
target = 'Severity'

print(f"using features: {features}")

# drop NaN
df_clean = df[features + [target]].dropna()
print(f"cleaned rows: {len(df_clean)}")

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

print(f"train: {len(X_train)}, test: {len(X_test)}")

# train model
print("training...")
model = LogisticRegression(max_iter=1000, random_state=38)
model.fit(X_train, y_train)

# check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.3f}")

print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# see what features matter
print("feature weights:")
for i, feature in enumerate(features):
    print(f"{feature}: {model.coef_[0][i]:.4f}")

print("done")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('Training.csv')

# Separate features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Encode target if it's categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(rf_model, 'rf.pkl')
joblib.dump(le, 'label_encoder.pkl')


from sklearn.svm import SVC

# Train SVC model on the same data
svc_model = SVC(probability=True)
svc_model.fit(X_train, y_train)

# Save model
joblib.dump(svc_model, 'svc.pkl')

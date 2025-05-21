import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import shap

# 1. loading
data = pd.read_csv(f'.csv')
self_esteem = data.iloc[:, -1]
low_self_esteem = (self_esteem < 31).astype(int)
features = data.iloc[:, 1:-1]
print(np.average(low_self_esteem))

# 2. Splitting data
X_train, X_test, y_train, y_test = train_test_split(features, low_self_esteem, test_size=0.2, random_state=42)

# 3. Modeling pipeline
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, probability=True))
model.fit(X_train, y_train)

# 4. Custom prediction function (ensure column names)
def model_predict(X_array):
    X_df = pd.DataFrame(X_array, columns=features.columns)  # Explicit column names
    return model.predict_proba(X_df)

# 5. SHAP explainer
background = shap.sample(X_train, 100, random_state=42)
explainer = shap.KernelExplainer(model_predict, background)

# 6. Explain the first 24 samples
X_eval = X_test.iloc[:24]
shap_values = explainer.shap_values(X_eval, nsamples=1000)

# Select the SHAP value for the low self-esteem class (class 1)
shap_low = shap_values[:, :, 1]  # (24, 96)
shap_df = pd.DataFrame(shap_low, columns=features.columns)
shap_df.to_csv('shap_values.csv', index=False)

# visualization
shap.summary_plot(shap_low, X_eval, feature_names=features.columns, max_display=20)
shap.summary_plot(shap_low, X_eval, feature_names=features.columns, plot_type='bar', max_display=20)

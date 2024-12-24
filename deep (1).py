import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Charger les données
data = pd.read_csv('data.csv')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
#categorical_cols=['Contract','Churn','Dependents','Partner']
categorical_cols = ['Partner','Churn', 'Contract', 'Dependents', 'TechSupport', 'StreamingMovies', 'StreamingTV']
specific_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']

data_std = pd.DataFrame(StandardScaler().fit_transform(data[specific_cols]).astype('float64'), columns=specific_cols)

data.drop(columns=specific_cols, inplace=True)
data = pd.concat([data, data_std], axis=1)

print("Avant la suppression des lignes avec des valeurs nulles dans TotalCharges:", data.shape)

# Supprimer les lignes avec des valeurs nulles dans la colonne "TotalCharges"
data.dropna(subset=['TotalCharges'], inplace=True)

# Encoder les variables catégorielles
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])
categorical_cols = ['Partner','Contract', 'Dependents', 'TechSupport', 'StreamingMovies', 'StreamingTV']
#categorical_cols=['Contract','Dependents','Partner']

# Séparer les features (X) et le label (y)
X = data[categorical_cols + specific_cols]
y = data['Churn']
print(X.columns)
print(X)
print(X.shape)
print(y)
print(y.shape)

# Appliquer SMOTE aux données
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Diviser les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Construction du modèle TabNet
model = TabNetClassifier()

# Entraînement du modèle
model.fit(
    X_train.values, y_train,
    eval_set=[(X_test.values, y_test)],
    patience=50,
    max_epochs=150
)

# Prédiction sur l'ensemble de test
y_pred_classes = model.predict(X_test.values)

# Calcul de la précision
accuracy = (y_pred_classes == y_test).mean()
print("Test Accuracy:", accuracy)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Afficher la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred_classes))

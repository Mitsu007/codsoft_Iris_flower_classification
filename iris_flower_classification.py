import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_path = r'C:\Users\mitus\OneDrive\Desktop\mitu intern\code soft\IRIS.csv'

iris_df = pd.read_csv(data_path, encoding='latin1')

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris_df[features]
y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

model_accuracy = accuracy_score(y_test, y_pred)

def predict_species():
    """Prompts the user for input and predicts the Iris species."""
    print("\nIRIS Species Prediction")
    try:
        sepal_length = float(input("Enter Sepal Length (cm): "))
        sepal_width = float(input("Enter Sepal Width (cm): "))
        petal_length = float(input("Enter Petal Length (cm): "))
        petal_width = float(input("Enter Petal Width (cm): "))

        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]], 
            columns=features
        )

        predicted_species = rf_model.predict(input_data)
        print(f"\nPredicted Iris species: {predicted_species[0]}")
        print(f"Model Accuracy: {model_accuracy * 100:.2f}%")
    except ValueError:
        print("\nInvalid input. Please enter numerical values for the features.")

predict_species()
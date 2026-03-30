import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

try:
    # 1. Dataset Loading
    df = pd.read_csv('tox21.csv')
    print("--- Dataset Loaded Successfully ---")

    # 2. Data Cleaning & Preprocessing
    df = df.fillna(0)
    
    # Selecting Features and Target
    # Dropping non-numeric or irrelevant columns
    X = df.drop(columns=['mol_id', 'smiles', 'NR-AR'], errors='ignore')
    y = df['NR-AR']

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Building - Random Forest Classifier
    print("Training the updated model... please wait.")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # 5. Model Prediction & Scoring
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Final Results ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Toxic', 'Toxic'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Toxicity Prediction Confusion Matrix\nAccuracy: {acc*100:.2f}%')

    # Saving the output graph
    plt.savefig('result_graph.png')
    print("\n[SUCCESS] Graph 'result_graph.png' ke naam se save ho gaya hai.")
    plt.show()

except Exception as e:
    print(f"Error: {e}")

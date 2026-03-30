import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


try:
    
    df = pd.read_csv('tox21.csv')
    print("--- Dataset Loaded Successfully ---")
    
    
    df = df.fillna(0)

    
    
    X = df.drop(columns=['mol_id', 'smiles', 'NR-AR'], errors='ignore') 
    y = df['NR-AR'] 

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    
    print("Training the updated model... please wait.")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Final Results ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Toxic', 'Toxic'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Toxicity Prediction Confusion Matrix\nAccuracy: {acc*100:.2f}%')
    
    
    plt.savefig('result_graph.png')
    print("\n[SUCCESS] Graph 'result_graph.png' ke naam se save ho gaya hai.")
    plt.show()

except Exception as e:
    print(f"Error: {e}")

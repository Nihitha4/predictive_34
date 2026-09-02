import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def main():
    df = pd.read_csv('data/churn_data.csv')

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print('Accuracy:', accuracy_score(y_test, pred))
    print('ROC AUC:', roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred))


if __name__ == '__main__':
    main()

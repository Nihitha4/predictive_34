import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main():
    df = pd.read_csv('data/creditcard.csv')

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print('Confusion matrix:\n', confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print('ROC AUC:', roc_auc_score(y_test, pred))


if __name__ == '__main__':
    main()

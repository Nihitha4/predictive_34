import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model_name, model):
    df = pd.read_csv('data/house_prices.csv')
    X = df.drop(columns=['price'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print(f'{model_name} RMSE:', mean_squared_error(y_test, pred, squared=False))
    print(f'{model_name} R2:', r2_score(y_test, pred))


def main():
    evaluate_model('Ridge', Ridge(alpha=1.0))
    evaluate_model('Lasso', Lasso(alpha=0.1, max_iter=10000))


if __name__ == '__main__':
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    df = pd.read_csv('data/car_data.csv')

    X = df.drop(columns=['price'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(f'{name} RMSE:', mean_squared_error(y_test, pred, squared=False))
        print(f'{name} R2:', r2_score(y_test, pred))


if __name__ == '__main__':
    main()

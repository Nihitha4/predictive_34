import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # Load dataset
    df = pd.read_csv('data/student_performance.csv')

    # Split features and target
    X = df.drop(columns=['performance'])
    y = df['performance']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model and tuning
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    grid = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)

    print('Best parameters:', grid.best_params_)
    print('MAE:', mean_absolute_error(y_test, pred))
    print('RMSE:', mean_squared_error(y_test, pred, squared=False))
    print('R2:', r2_score(y_test, pred))


if __name__ == '__main__':
    main()

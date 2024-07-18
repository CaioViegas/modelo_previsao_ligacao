import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from preprocessor import preprocessamento

def modelo_otimizado(dataset, target):
    dataset = preprocessamento(dataset)

    X = dataset.drop(target, axis=1)
    y = dataset[target]

    rfc = RandomForestClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('pca', PCA(n_components=10)),
        ('selector', SelectKBest(f_classif, k=10)),
        ('model', rfc)  
    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f'Melhores parâmetros encontrados:')
    print(best_model.get_params())

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(best_model)
    
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross-Validation Accuracy: {cross_val_scores.mean():.2f}')
    print(f'Acurácia: {accuracy:.2f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')

    joblib.dump(best_model, "melhor_modelo.joblib")  

    return best_model

if __name__ == "__main__":
    df = pd.read_csv("./data/dataset.csv")
    modelo_otimizado(df, "conversion_status")
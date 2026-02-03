from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def treinar_modelo_tempo_gasto(df):
    print("Iniciando o treinamento do modelo de tempo gasto...")

    # variaveis preditoras
    features = ["age", "gender_n", "income_level_n", "hobbies_count"]
    X = df[features]
    
    # Target
    y = df["time_spent"]

    # Divisão entre Treino e Teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regressão
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avaliação Erro médio
    preds = model.predict(X_test)
    erro_medio = mean_absolute_error(y_test, preds)
    precisao = r2_score(y_test, preds)
    
    return model, features, erro_medio, precisao

def treinar_modelo_conteudo(df):
    print("Iniciando o treinamento do modelo de conteudo...")
    
    # variaveis preditoras
    features = ["age", "gender_n", "income_level_n", "hobbies_count"]
    X = df[features]
    
    # target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["content_type_preference"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_target
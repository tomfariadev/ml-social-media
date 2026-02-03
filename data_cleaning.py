import pandas as pd
from sklearn.preprocessing import LabelEncoder

def cleaning(df):
    df = df.sort_values(by="country")
    # Transformar texto em números
    le = LabelEncoder()
    df["gender_n"] = le.fit_transform(df["gender"])
    df["income_level_n"] = le.fit_transform(df["income_level"])
    # categorizar 
    df["time_spent"] = df["time_on_feed_per_day"] + df["time_on_explore_per_day"] + df["time_on_messages_per_day"] + df["time_on_reels_per_day"]
    df["usage_Level"] = pd.cut(df["time_spent"], bins=[0, 120, 300, 1000], labels=[0, 1, 2])

    return df
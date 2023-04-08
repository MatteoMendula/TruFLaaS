import utils
import tensorflow as tf
import custom_extension

if __name__ == "__main__":
    #initialize global model

    import pandas as pd

    df = pd.read_pickle("./data/df.pkl")
    df["type"].value_counts()
    df_only_data = df.copy()
    df_only_data.drop(['type'], axis = 1, inplace = True) 
    print(df)
    print(df_only_data)

    print(df.shape)
    print(df_only_data.shape)

    import numpy as np
    mu, sigma = 0, 0.5
    noise = np.random.normal(mu, sigma, df_only_data.shape) 

    print(noise.shape)
    noisy_df = df_only_data + noise
    print(noisy_df)


    # noisy_df.insert(0, "type", list(df["type"])) 
    noisy_df["type"] = df["type"]
    print(df["type"].value_counts())
    print(noisy_df["type"].value_counts())

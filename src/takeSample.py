import pandas as pd

df = pd.read_csv('data/raw/US_Accidents_March23.csv')
sample_df = df.sample(n=100000, random_state=21)
sample_df.to_csv('data/processed/sample_output.csv', index=False)
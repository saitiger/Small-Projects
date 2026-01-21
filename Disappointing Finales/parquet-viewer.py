import pandas as pd
df = pd.read_parquet("Disappointing Finales/tv-finale-pipeline/data/processed/shows.parquet")
# print(df.head())
# print(df.columns)
# print(df.iloc[0])

# Missing Values
# print(df.isna().sum()) # 22 shows have missing value in 'end_year'
missing_end_year = df[df['end_year'].isna()]['title']
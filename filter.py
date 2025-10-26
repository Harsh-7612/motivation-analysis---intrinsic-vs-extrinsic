import pandas as pd

df = pd.read_csv('labelled.csv')
df = df[df['confidence']>0.88].reset_index(drop='True')
df.to_csv('pseudo_lbd.csv')


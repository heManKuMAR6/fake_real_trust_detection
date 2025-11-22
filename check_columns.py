import pandas as pd

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

print("TRUE COLUMNS:", true_df.columns)
print("FAKE COLUMNS:", fake_df.columns)

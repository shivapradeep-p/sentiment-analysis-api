import pandas as pd
# Load the data from the dataset into pandas dataframe
df = pd.read_csv("Tweets.csv")

# Display the first 5 rows to verify the data has been loaded correctly
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display the number of rows and columns in the dataset
print("\nDataset Shape:", df.shape)

# Display the coloumn name, no of null values and data types
print("\nDataset Info:")
print(df.info())

# Show the distribution and scount of sentiment labels([positive, negative, neutral])
print("\nSentiment labels and counts:")
print(df["airline_sentiment"].value_counts())
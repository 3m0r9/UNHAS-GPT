import pandas as pd

# Read data from CSV file
df = pd.read_csv('converted_metadata.csv')

# Remove ":443" from the URLs
df['PDF_url'] = df['PDF_url'].str.replace('id:443', 'id')

# Save the updated DataFrame to the same file
df.to_csv('converted_metadata2.csv', mode='a', header=False, index=False)

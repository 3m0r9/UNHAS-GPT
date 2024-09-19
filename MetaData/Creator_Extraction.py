import pandas as pd

# Load the data from the CSV file into a DataFrame
df = pd.read_csv("metadata_JSON_URL_filtered.csv")

# Extract the creators' names

def extract_creator_names(row):
    creators = eval(row)
    if creators is None:
        return ""
    elif isinstance(creators, list) and len(creators) > 0:
        creator = creators[0]
        if "name" in creator:
            name = creator["name"]
            given_name = name.get("given", "")
            family_name = name.get("family", "")
            if given_name is None or family_name is None:
                return ""
            return given_name + " " + family_name
    return ""

creators_names = df["creators"].apply(extract_creator_names)


# Add the new_creator column to the DataFrame
df["new_creator"] = creators_names

# Save the updated DataFrame to the CSV file
df.to_csv("metadata_JSON_URL_final.csv", index=False)

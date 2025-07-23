import pandas as pd

# Assume these DataFrames and variables are defined elsewhere in your code
# For demonstration, let's create some sample data:

# Sample for lang_df
lang_name = "french" # Or "german", "spanish", etc.
lang_column = "french_text" # Or "german_column", etc.

lang_df = pd.DataFrame({
    'french_text': ["Bonjour", "Comment ça va?", "Merci"],
    'id': [1, 2, 3],
    'other_data': ['a', 'b', 'c']
})

# Sample for eng_df
eng_column = "english_text" # Or "eng_column_alt", etc.

eng_df = pd.DataFrame({
    'english_text': ["Hello", "How are you?", "Thank you"],
    'id': [1, 2, 3],
    'text': ["Hi", "Howdy", "Cheers"] # This 'text' column is important for the 'else' case
})


# The line you provided:
df = pd.DataFrame({
    lang_name.lower(): lang_df[lang_column] if lang_column in lang_df else lang_df['text'],
    'english': eng_df[eng_column] if eng_column in eng_df else eng_df['text']
})

print(df)
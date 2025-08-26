import unicodedata

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        # Check if the character's category starts with 'C' (for Control)
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            # Escape the control character to its Unicode representation
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

# Here is a string with visible characters and invisible control characters
s = "Hello\tWorld!\nThis is a test.\x00"

# Now, let's see what the function does to the string
# The function will replace the control characters (\t, \n, \x00) with their
# escaped Unicode representations
escaped_s = replace_control_characters(s)

print(f"Original String: '{s}'")
print(f"Processed String: '{escaped_s}'")

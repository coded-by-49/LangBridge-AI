import pandas as pd
with open("data/Jw300/jw300.ig.txt", "r", encoding="utf-8") as f:
    igbo_lines = [line.strip() for line in f]  # make each line a list item 
with open("data/Jw300/jw300.en.txt", "r", encoding="utf-8") as f:
    english_lines = [line.strip() for line in f] 

assert len(igbo_lines) == len(english_lines), f"unequivalent lines between the text files"
data = {"igbo": igbo_lines, "english": english_lines}
eng_igbo_pair = pd.DataFrame(data)

eng_igbo_pair.to_csv("data/Jw300/ig_en.csv", index=False)
print(f"Saved {len(eng_igbo_pair)} Igbo-English pairs to data/jw300/ig_en.csv")

print(eng_igbo_pair.head())

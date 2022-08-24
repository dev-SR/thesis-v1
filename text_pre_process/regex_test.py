import re
import os
text = "sociation for omputational inguistics pages ancouver anada uly ugust . c ss"
clean_text_files = [os.path.join(os.getcwd(
), "data\\papers\\text_demo", f) for f in os.listdir("data/papers/text_demo") if f.endswith("clean.txt")]

for f in clean_text_files:
    with open(f, "r", encoding="utf-8") as d:
        text = d.read()
        text = re.sub(r'\s\.', '.', text)
        print(text)
    break

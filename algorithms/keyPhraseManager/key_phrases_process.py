import pandas as pd


def strip_key_phrases(row):
    if row['key_phrases'] is not None:
        return row['key_phrases'].strip()


def saveKeyPhrases(df_path, key_phrases, paper_id):
    try:
        # existing_df = pd.read_csv(df_path)
        # already_exits = existing_df[existing_df['paper_id']
        #                             == paper_id].shape[0]
        # if already_exits:
        #     return 0
        df = pd.DataFrame(
            key_phrases, columns=['key_phrases', 'weight'])

        df['key_phrases'] = df.apply(strip_key_phrases, axis=1)
        # add paper_id column
        paper_ids = [paper_id] * len(df)
        df.insert(0, 'paper_id', paper_ids)
        df.to_csv(df_path, mode='a', index=False, header=False)
        return 1
    except Exception:
        # open("keyphrase.txt", 'a').write(paper_id).close()
        print(f"Error saving key phrases{paper_id}")


def saveFailedInKeyPhrasesExtraction(df_path, paper_id):
    df = pd.DataFrame([paper_id])
    df.to_csv(df_path, mode='a', index=False, header=False)

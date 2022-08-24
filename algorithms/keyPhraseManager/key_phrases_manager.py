import pandas as pd


class KeyPhrasesManager:
    def __init__(self, csv_path):
        self.key_phrases_df = pd.read_csv(csv_path)

    def getKeyPhrasesList(self, paper_id):
        records = self.key_phrases_df[self.key_phrases_df['paper_id'] == paper_id][[
            'key_phrases', 'weight']].to_records(index=False)

        return list(records)

    def isKeyPhraseExist(self, paper_id):
        return paper_id in self.key_phrases_df['paper_id'].values

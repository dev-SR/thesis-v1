
import pke
"""
!pip install git+https://github.com/boudinfl/pke.git
!python -m spacy download en_core_web_sm
 """


def getWeightedKeyPhrasesUsingYAKE(paper_text_full_path, Top_n=15):
    with open(paper_text_full_path, "r", encoding="utf-8") as f:
        text = f.read()
        print("YAKE")
        if len(text) > 1000000 - 1:
            text = text[:1000000 - 1]
        # initialize keyphrase extraction model, here TopicRank
        extractor = pke.unsupervised.YAKE()

        # load the content of the document, here document is expected to be a simple
        # test string and preprocessing is carried out using spacy
        extractor.load_document(input=text, language='en')

        # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
        # and adjectives (i.e. `(Noun|Adj)*`)
        extractor.candidate_selection()

        # candidate weighting, in the case of TopicRank: using a random walk algorithm
        extractor.candidate_weighting()

        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        keyphrases = extractor.get_n_best(n=Top_n)
        return keyphrases

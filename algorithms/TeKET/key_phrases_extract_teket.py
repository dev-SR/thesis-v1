import nltk
import math
import re
import itertools
import string
from nltk.tokenize import word_tokenize
import os
import operator
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
ps = PorterStemmer()


class Node:
    def __init__(self, word, tfidf):  # creat constructor
        self.word = word
        self.mu = 0
        self.next = None
        self.prev = None
        self.TFIDF = tfidf
        self.root_status = False


class Tree:

    root = None

    #
    # This method adds a node in the tree to a particular position based on its word's appearance in the
    # key phrase
    #

    def AddNode(self, word, phrase, wordIndex, rootIndex, tfidf):
        if word is None:
            print("AddNode() :: word is None")
            return

        if self.root == None:
            if len(word) < 3:
                return False
            else:
                self.root = Node(word, tfidf)
                self.root.root_status = True
                return True

        depth = abs(wordIndex - rootIndex)
        count = 0
        newNode = self.root
        if wordIndex < rootIndex:
            while count < depth - 1 or newNode != None:
                if newNode.word == word:
                    return True

                elif newNode.prev != None and newNode.prev.word == phrase[rootIndex - count - 1]:
                    newNode = newNode.prev
                    count = count + 1
                    continue

                elif newNode.next != None and newNode != self.root and newNode.next.word == phrase[rootIndex - count - 1]:
                    newNode = newNode.next
                    count = count + 1
                    continue

                else:
                    if newNode.prev == None:
                        newNode.prev = Node(word, tfidf)
                        return True

                    elif newNode.next == None and newNode != self.root:
                        newNode.next = Node(word, tfidf)
                        return True

                    elif newNode.prev != None and newNode.prev.word != phrase[rootIndex - count - 1]:
                        if newNode.prev.TFIDF < tfidf:
                            newNode.prev = None
                            newNode.prev = Node(word, tfidf)
                            return True
                        else:
                            return False

                    elif newNode.next != None and newNode != self.root and newNode.next.word != phrase[rootIndex - count - 1]:
                        if newNode.next.TFIDF < tfidf:
                            newNode.next = None
                            newNode.next = Node(word, tfidf)
                            return True
                        else:
                            return False

                    else:
                        return False

        if wordIndex > rootIndex:

            while count < depth - 1 or newNode != None:
                if newNode.word == word:
                    return True

                elif newNode.next != None and newNode.next.word == phrase[rootIndex + count + 1]:
                    newNode = newNode.next
                    count = count + 1
                    continue

                elif newNode.prev != None and newNode != self.root and newNode.prev.word == phrase[rootIndex + count + 1]:
                    newNode = newNode.prev
                    count = count + 1
                    continue

                else:
                    if newNode.next == None:
                        newNode.next = Node(word, tfidf)
                        return True

                    elif newNode.prev == None and newNode != self.root:
                        newNode.prev = Node(word, tfidf)
                        return True

                    elif newNode.next != None and newNode.next.word != phrase[rootIndex + count + 1]:
                        if newNode.next.TFIDF < tfidf:
                            newNode.next = None
                            newNode.next = Node(word, tfidf)
                            return True
                        else:
                            return False

                    elif newNode.prev != None and newNode != self.root and newNode.prev.word != phrase[rootIndex + count + 1]:
                        if newNode.prev.TFIDF < tfidf:
                            newNode.prev = None
                            newNode.prev = Node(word, tfidf)
                            return True
                        else:
                            return False
                    else:
                        return False

            return False

    def DecreaseValuesOfASubTree(self, node):
        if node:
            self.DecreaseValuesOfASubTree(node.prev)
            node.mu -= 1
            self.DecreaseValuesOfASubTree(node.next)
            return

    def UpdateMuValues(self, phrase):
        rootPosition = -1

        for i in range(0, len(phrase), 1):
            if self.root.word == phrase[i]:
                rootPosition = i

        if rootPosition < 0:
            print("WARNING: root is not found in phrase")
            return
        elif rootPosition == 0 and self.root.prev is not None:
            self.DecreaseValuesOfASubTree(self.root.prev)
        elif rootPosition == len(phrase) - 1 and self.root.next is not None:
            self.DecreaseValuesOfASubTree(self.root.next)

        self.root.mu += 1
        newNode = self.root.prev

        for i in range(rootPosition - 1, -1, -1):
            if newNode == None:
                break

            if newNode.word == phrase[i]:
                newNode.mu += 1

                if newNode.prev != None and i - 1 > -1 and newNode.prev.word == phrase[i - 1]:
                    if newNode.next != None:
                        self.DecreaseValuesOfASubTree(newNode.next)

                    newNode = newNode.prev
                    continue

                elif newNode.next != None and i - 1 > -1 and newNode.next.word == phrase[i - 1]:
                    if newNode.prev != None:
                        self.DecreaseValuesOfASubTree(newNode.prev)

                    newNode = newNode.next
                    continue

                else:
                    self.DecreaseValuesOfASubTree(newNode.prev)
                    self.DecreaseValuesOfASubTree(newNode.next)
                    break
            else:
                self.DecreaseValuesOfASubTree(newNode)

        newNode = self.root.next

        for i in range(rootPosition + 1, len(phrase), 1):
            if newNode == None:
                break

            if newNode.word == phrase[i]:
                newNode.mu += 1

                if newNode.next != None and i + 1 < len(phrase) and newNode.next.word == phrase[i + 1]:
                    if newNode.prev != None:
                        self.DecreaseValuesOfASubTree(newNode.prev)

                    newNode = newNode.next
                    continue

                elif newNode.prev != None and i + 1 < len(phrase) and newNode.prev.word == phrase[i + 1]:
                    if newNode.next != None:
                        self.DecreaseValuesOfASubTree(newNode.next)

                    newNode = newNode.prev
                    continue

                else:
                    self.DecreaseValuesOfASubTree(newNode.prev)
                    self.DecreaseValuesOfASubTree(newNode.next)
                    break
            else:
                self.DecreaseValuesOfASubTree(newNode)

    def TreePruning(self, node, k):
        if node is None:
            return

        if node.mu < k:
            node = None
            return

        if node.prev is not None and node.prev.mu < k:
            node.prev = None

        self.TreePruning(node.prev, k)

        if node.next is not None and node.next.mu < k:
            node.next = None

        self.TreePruning(node.next, k)

    def RootToLeafPaths(self, node, nl, ln, rn):
        if node is None:
            return

        nl.append(node)
        self.RootToLeafPaths(node.prev, nl, ln, rn)
        if node.prev is None and node.next is None:
            x = []
            for j in range(0, len(nl), 1):
                x.append(nl[j])
            if len(nl) > 1:
                if nl[0].prev == nl[1]:
                    ln.append(x)
                else:
                    rn.append(x)
            else:
                # If nl contains only one node then it would be appended in the left node list, i.e., ls in this code
                ln.append(x)
                return
        self.RootToLeafPaths(node.next, nl, ln, rn)
        nl.pop()

    def FindPhaseInCandidatePhrases(self, node_list, candidate_phrases):
        phrase = self.GetPhrase(node_list)

        exist = False

        if phrase in candidate_phrases:
            exist = True
            return exist

        for i in range(0, len(candidate_phrases), 1):
            if phrase in candidate_phrases[i]:
                exist = True
                break

        return exist

    def FindPhaseExistanceInList(self, node_list, final_node_list):
        ph = self.GetPhrase(node_list)

        exist = False

        for i in range(0, len(final_node_list), 1):
            fph = self.GetPhrase(final_node_list[i])
            if ph == fph:
                exist = True
                break

        return exist

    def FindNodeListToExtractKeyPhrases(self, candidate_phrases, mu, final_node_list):
        self.TreePruning(self.root, mu)
        if self.root is None:
            return final_node_list
        elif self.root.mu < mu:
            return final_node_list

        nl = []  # Total node list
        ln = []  # Left node list
        rn = []  # Right node list

        self.RootToLeafPaths(self.root, nl, ln, rn)

        x = []
        x.append(self.root)
        if self.FindPhaseExistanceInList(x, final_node_list) is False:
            final_node_list.append(x)
        left_phrase_list = []
        right_phrase_list = []
        for i in range(0, len(ln), 1):
            newPhrase = []
            for l in range(0, len(ln[i])):
                newPhrase.insert(0, ln[i][l])
                morePhrase = []
                morePhrase = newPhrase.copy()
                left_phrase_list.append(morePhrase)
                if self.FindPhaseExistanceInList(morePhrase, final_node_list) is False and self.FindPhaseInCandidatePhrases(morePhrase, candidate_phrases) is True:
                    final_node_list.append(morePhrase)

        for j in range(0, len(rn), 1):
            newPhrase = []
            for m in range(0, len(rn[j]), 1):
                morePhrase = []
                newPhrase.append((rn[j])[m])
                morePhrase = newPhrase.copy()

                right_phrase_list.append(morePhrase)

                if self.FindPhaseExistanceInList(morePhrase, final_node_list) is False and self.FindPhaseInCandidatePhrases(morePhrase, candidate_phrases) is True:
                    final_node_list.append(morePhrase)
        #
        # Following, we do conccatenation of two sets of strings
        # Details could be found from the following wiki link.
        # https://en.wikipedia.org/wiki/Concatenation
        #
        # Manual concatenation is done since the list contains objects
        #

        for i in range(1, len(left_phrase_list)):
            newPhrase = []
            if len(left_phrase_list[i]) > 1:
                for l in range(0, len(left_phrase_list[i])):
                    newPhrase.append(left_phrase_list[i][l])
            else:
                continue
            for j in range(0, len(right_phrase_list)):
                if len(right_phrase_list[j]) > 1:
                    morePhrase = []
                    morePhrase = newPhrase.copy()
                    for r in range(1, len(right_phrase_list[j])):
                        morePhrase.append(right_phrase_list[j][r])

                    if self.FindPhaseExistanceInList(morePhrase, final_node_list) is False and self.FindPhaseInCandidatePhrases(morePhrase, candidate_phrases) is True:
                        final_node_list.append(morePhrase)

        return final_node_list

    def GetRoot(self):
        return self.root

    def PrintTree(self, root):
        if root:
            self.PrintTree(root.prev)
            print(root.word + " " + str(root.mu))
            self.PrintTree(root.next)

    def GetPhrase(self, node_list):
        s = ""
        if len(node_list) > 0:
            for i in range(0, len(node_list)):
                if i != len(node_list) - 1:
                    if node_list[i] is not None:
                        s += str(node_list[i].word) + " "
                else:
                    if node_list[i] is not None:
                        s += str(node_list[i].word)
        else:
            print("WARNING: Phrase is empty")

        return s

    def CheckPhraseExist(self, phrase, candidate_phrases):
        ph = self.GetPhrase(phrase)
        if len(ph) > 0:
            for i in range(0, len(candidate_phrases)):
                if ph == candidate_phrases[i]:
                    return True
            return False
        else:
            return False


class TreeManager:
    def FindTFIDFScore(self, word, TFIDF_tuple):
        for i in range(0, len(TFIDF_tuple), 1):
            if word == TFIDF_tuple[i][0]:
                return TFIDF_tuple[i][1]
        return 0

    def ProcessCandidatePhrase(self, TFIDF_tuple, candidate_phrases, mu, final_node_list):

        if TFIDF_tuple is None:
            print("WARNING: ProcessCandidatePhrase():: TFIDF_tuple is None")
            return

        if candidate_phrases is None:
            print("WARNING: ProcessCandidatePhrase():: candidate_phrases is None")
            return

        for j in range(0, len(TFIDF_tuple), 1):
            word = TFIDF_tuple[j][0]
            score = TFIDF_tuple[j][1]

            tree = Tree()

            tree.AddNode(word, None, 0, 0, score)

            for k in range(0, len(candidate_phrases), 1):
                index = -1
                phrase = candidate_phrases[k].split()
                if tree.root.word in phrase:
                    index = phrase.index(tree.root.word)

                if index == -1:
                    continue
                else:
                    for l in range(index - 1, -1, -1):
                        if tree.AddNode(phrase[l], phrase, l, index, self.FindTFIDFScore(phrase[l], TFIDF_tuple)) is False:
                            break
                    for m in range(index + 1, len(phrase), 1):
                        if tree.AddNode(phrase[m], phrase, m, index, self.FindTFIDFScore(phrase[m], TFIDF_tuple)) is False:
                            break

                    tree.UpdateMuValues(phrase)

            tree.FindNodeListToExtractKeyPhrases(
                candidate_phrases, mu, final_node_list)

    def PrintKeyPhrases(self, final_node_list):

        tuple_list = []

        for i in range(0, len(final_node_list)):
            s = ""
            for j in range(0, len(final_node_list[i])):
                s += str((final_node_list[i])[j].word) + " "

            tuple_list.append(tuple((s, (final_node_list[i])[j].TFIDF)))

        tuple_list.sort(key=operator.itemgetter(1), reverse=True)

        print(tuple_list)

    def ReturnKeyPhrases(self, final_node_list, candidate_phrases, text, alpha=1, beta=1, gamma=1):

        tuple_list = []

        TFIDF_list = []
        mu_list = []

        PD = 0
        for i in range(0, len(candidate_phrases), 1):
            if len(candidate_phrases[i].split()) > 0:
                PD += 1

        for i in range(0, len(final_node_list), 1):
            x = 0
            y = 0

            for j in range(0, len(final_node_list[i])):
                x += final_node_list[i][j].TFIDF
                y += final_node_list[i][j].mu
            TFIDF_list.append(x)
            mu_list.append(y)

        text_split = text.split()

        for i in range(0, len(final_node_list), 1):
            s = ""
            tfidf_score = 0
            mu_score = 0
            ava_TF = 0
            count = 0
            entropy = 0
            root_mu_value = 0
            relevance_score = 0

            for j in range(0, len(final_node_list[i])):
                if final_node_list[i][j].root_status is True:
                    root_mu_value = final_node_list[i][j].mu

            for j in range(0, len(final_node_list[i])):
                s += str((final_node_list[i])[j].word) + " "

                if final_node_list[i][j].TFIDF > 0:
                    entropy += (final_node_list[i][j].TFIDF *
                                (math.log2(1 / final_node_list[i][j].TFIDF)))
                else:
                    entropy += 0

                tfidf_score += final_node_list[i][j].TFIDF
                count += 1

                mu_score += final_node_list[i][j].mu

                ava_TF += text_split.count(
                    final_node_list[i][j].word) / len(text_split)

                if root_mu_value > 0:
                    relevance_score += final_node_list[i][j].mu / root_mu_value
                else:
                    relevance_score += 0

            temp_tuple = tuple([s, tfidf_score * mu_score])
            found = False
            for k in range(0, len(tuple_list)):
                if tuple_list[k][0] == s:
                    found = True
                    break

            if found is False:
                tuple_list.append(temp_tuple)

        tuple_list.sort(key=operator.itemgetter(1), reverse=True)

        return tuple_list


class ExtractCandidate:

    def __init__(self, text, select=1):
        self.text = text

        if select == 1:
            self.extract_candidate_chunks()

    def extract_candidate_chunks(self):
        grammar = r'KT: { (<NN.*>+ <JJ.*>?)|(<JJ.*>? <NN.*>+)}'

        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        # tokenize, POS-tag, and chunk using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(
            sent) for sent in nltk.sent_tokenize(self.text))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(
            chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
        # join constituent chunk words into a single chunked phrase
        candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(
            all_chunks, lambda word__pos__chunk: word__pos__chunk[2] != 'O') if key]

        x = [cand for cand in candidates if cand not in stop_words and not all(
            char in punct for char in cand)]

        data = []

        for i in range(0, len(x), 1):
            if len(x[i].split()) == 1:
                if re.match("^[A-Za-z0-9]*$", x[i]):
                    if len(x[i]) > 2:
                        data.append(x[i])

            else:
                add = ""
                split = x[i].split()
                lenth = len(split)
                for i in range(0, lenth, 1):
                    king = re.match("^[A-Za-z0-9]*$", split[i])
                    if len(str(king)) > 2:
                        add = add + " " + split[i]
                data.append(add.strip())

        return data

    def CleaningCandidatePhrases(self, probable_phrases):
        probpr = []

        for i in range(0, len(probable_phrases), 1):
            words = word_tokenize(probable_phrases[i])
            wa = []
            for w in words:
                x = ps.stem(w)
                wa.append(x)
            probable_phrases[i] = (' '. join(wa))

        for i in range(0, len(probable_phrases), 1):
            for j in range(0, len(probable_phrases[i]), 1):
                regex = re.compile('[@_!#$%^&*()<>?/\|}{~:].')
                if regex.search(probable_phrases[i]) == None:
                    probpr.append(probable_phrases[i])
                    break
                else:
                    continue

        for i in range(0, len(probable_phrases), 1):
            temp_probable_phrases = probable_phrases[i].split(" ")
            min_length = True
            for k in range(0, len(temp_probable_phrases), 1):
                if len(temp_probable_phrases[k]) < 2:
                    min_length = False

            if min_length:
                count = 0
                for j in range(0, len(probable_phrases), 1):
                    if probable_phrases[i] == probable_phrases[j]:
                        count = count + 1

                    if count > 2:
                        probpr.append(probable_phrases[i])
                        break
            else:
                continue

        return probpr

    def CandidatePhraseHandler(self):
        candidate_phrases = self.extract_candidate_chunks()
        # print(candidate_phrases[:10])
        candidate_phrases = self.CleaningCandidatePhrases(candidate_phrases)
        # print(candidate_phrases[:10])
        return candidate_phrases


class NounTFCalculation:

    def NounTF(self, candidate_phrases):

        NounTF_tuple_list = []
        Noun_word = []

        NounTF_chack = []

        for i in range(0, len(candidate_phrases), 1):
            word_tokenize = nltk.word_tokenize(candidate_phrases[i])
            tagged_token = nltk.pos_tag(word_tokenize)

            for j in range(0, len(tagged_token), 1):
                if tagged_token[j][1] == 'NN' or 'NNS' or 'NNP' or 'NNPS':
                    Noun_word.append(tagged_token[j][0])

        for i in range(0, len(candidate_phrases), 1):
            word_tokenize = nltk.word_tokenize(candidate_phrases[i])
            tagged_token = nltk.pos_tag(word_tokenize)

            for j in range(0, len(tagged_token), 1):
                if tagged_token[j][1] == 'NN' or 'NNS' or 'NNP' or 'NNPS':
                    if len(tagged_token[j][0]) > 2:
                        NounTF_chack.append(tagged_token[j][0])
                        if Noun_word.count(tagged_token[j][0]) > 10:
                            if tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]))) not in NounTF_tuple_list:
                                NounTF_tuple_list.append(
                                    tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]))))
        return NounTF_tuple_list


def getWeightedKeyPhrasesUsingTeKET(paper_text_full_path, Top_n=15):
    treemanager = TreeManager()
    ntf = NounTFCalculation()
    mu = 2
    lsaf = 4
    with open(paper_text_full_path, "r", encoding="utf-8") as f:
        text = f.read()
        final_node_list = []
        # if len(text) > 1000000 - 1:
        #     text = text[:1000000 - 1]
        extract_candidate = ExtractCandidate(text)
        candidate_phrases = extract_candidate.CandidatePhraseHandler()
        candidate_phrases_count_lsaf = []
        for x in range(0, len(candidate_phrases), 1):
            if candidate_phrases.count(candidate_phrases[x]) > lsaf:
                candidate_phrases_count_lsaf.append(candidate_phrases[x])

        NounTF_tuple = ntf.NounTF(candidate_phrases_count_lsaf)
        treemanager.ProcessCandidatePhrase(
            NounTF_tuple, candidate_phrases_count_lsaf, mu, final_node_list)

        Keyphrases = treemanager.ReturnKeyPhrases(
            final_node_list, candidate_phrases_count_lsaf, text)

        return Keyphrases[:Top_n]


def getWeightedKeyPhrasesUsingTeKETReadText(text, Top_n=15):
    treemanager = TreeManager()
    ntf = NounTFCalculation()
    mu = 2
    lsaf = 4
    final_node_list = []
    # if len(text) > 1000000 - 1:
    #     text = text[:1000000 - 1]
    extract_candidate = ExtractCandidate(text)
    candidate_phrases = extract_candidate.CandidatePhraseHandler()
    candidate_phrases_count_lsaf = []
    for x in range(0, len(candidate_phrases), 1):
        if candidate_phrases.count(candidate_phrases[x]) > lsaf:
            candidate_phrases_count_lsaf.append(candidate_phrases[x])

    NounTF_tuple = ntf.NounTF(candidate_phrases_count_lsaf)
    treemanager.ProcessCandidatePhrase(
        NounTF_tuple, candidate_phrases_count_lsaf, mu, final_node_list)

    Keyphrases = treemanager.ReturnKeyPhrases(
        final_node_list, candidate_phrases_count_lsaf, text)

    # return Keyphrases[:Top_n]
    return Keyphrases


# print(getWeightedKeyPhrasesUsingTeKETReadText())

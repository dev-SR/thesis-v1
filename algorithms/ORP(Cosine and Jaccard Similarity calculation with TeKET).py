
# coding: utf-8

# In[1]:


from __future__ import division
import nltk
import math
import re
import mmap
import itertools
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
#from textblob import TextBlob as tb
import os
#import networkx
from nltk.stem.porter import *
import scipy.stats
import statistics
from statistics import mean
from statistics import stdev
import numpy as np
from nltk import ngrams
from os import walk
import operator

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#from porter2stemmer import Porter2Stemmer
#ps = Porter2Stemmer()

import numpy
import numpy as np

#from TFIDF import TFIDF
# from Remove_unexpected_char import remove_unexpected_char


# In[2]:


class Node:
    def __init__(self, word, tfidf): #creat constructor
        self.word = word
        self.mu = 0
        self.next = None
        self.prev = None
        self.TFIDF = tfidf
        self.root_status = False


# In[3]:


class Tree:

    root = None

    #
    # This method adds a node in the tree to a particular position based on its word's appearance in the
    # key phrase
    #

    def AddNode(self, word, phrase, wordIndex, rootIndex, tfidf):
        #phrase = phrase.split()

        #print("rootTFIDF...."+str(rootTFIDF))
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

        #print(word+"...."+str(self.root.word)+"..."+str(self.root.TFIDF) +"..."+str(tfidf))
        #print (phrase)

        if wordIndex < rootIndex:

            while count < depth - 1 or newNode != None:

                if newNode.word == word:
                    return True

                elif newNode.prev != None and newNode.prev.word == phrase[rootIndex - count -1]:
                    newNode = newNode.prev
                    count = count + 1
                    continue

                elif newNode.next != None and newNode != self.root and newNode.next.word == phrase[rootIndex - count -1]:
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

                    elif newNode.prev != None and newNode.prev.word != phrase[rootIndex - count -1]:
                        if newNode.prev.TFIDF < tfidf:
                            newNode.prev = None
                            newNode.prev = Node (word, tfidf)
                            return True
                        else:
                            return False

                    elif newNode.next != None and newNode != self.root and newNode.next.word != phrase[rootIndex - count -1]:
                        if newNode.next.TFIDF < tfidf:
                            newNode.next = None
                            newNode.next = Node (word, tfidf)
                            return True
                        else:
                            return False

                    else:
                        return False

        if wordIndex > rootIndex:

            while count < depth-1 or newNode != None:
                if newNode.word == word:
                    return True

                elif newNode.next != None and newNode.next.word == phrase[rootIndex + count +1]:
                    newNode = newNode.next
                    count = count + 1
                    continue

                elif newNode.prev != None and newNode != self.root and newNode.prev.word == phrase[rootIndex + count +1]:
                    newNode = newNode.prev
                    count = count + 1
                    continue

                else:
                    if newNode.next == None:
                        newNode.next = Node(word, tfidf)
                        return True

                    elif newNode.prev == None  and newNode != self.root:
                        newNode.prev = Node(word, tfidf)
                        #print(newNode.prev.word + " <-> " + word + " --- " + newNode.word)
                        return True

                    elif newNode.next != None and newNode.next.word != phrase[rootIndex + count + 1]:
                        if newNode.next.TFIDF < tfidf:
                            newNode.next = None
                            newNode.next = Node (word, tfidf)
                            return True
                        else:
                            return False

                    elif newNode.prev != None and newNode != self.root and newNode.prev.word != phrase[rootIndex + count + 1]:
                        if newNode.prev.TFIDF < tfidf:
                            newNode.prev = None
                            newNode.prev = Node (word, tfidf)
                            return True
                        else:
                            return False

                    else:
                        return False

            return False

    def DecreaseValuesOfASubTree (self, node):
        #return
        if node:
            self.DecreaseValuesOfASubTree (node.prev)
            node.mu -= 1
            self.DecreaseValuesOfASubTree (node.next)
            return

    def UpdateMuValues (self, phrase):
        #phrase = phrase.split()

        #later added code for testing
        #if len(phrase) == 1 and phrase == self.root.word:
            #return

        rootPosition = -1

        for i in range (0, len(phrase), 1):
            if self.root.word == phrase[i]:
                rootPosition = i

        if rootPosition < 0:
            print("WARNING: root is not found in phrase")
            return
        elif rootPosition == 0 and self.root.prev is not None:
            self.DecreaseValuesOfASubTree (self.root.prev)
        elif rootPosition == len(phrase) - 1 and self.root.next is not None:
            self.DecreaseValuesOfASubTree (self.root.next)

        self.root.mu += 1
        newNode = self.root.prev

        for i in range (rootPosition - 1, -1, -1):
            if newNode == None:
                break

            if newNode.word == phrase[i]:
                newNode.mu += 1

                if newNode.prev != None and i - 1 > -1 and newNode.prev.word == phrase[i - 1]:
                    if newNode.next != None :
                        self.DecreaseValuesOfASubTree (newNode.next)

                    newNode = newNode.prev
                    continue

                elif newNode.next != None and i - 1 > -1 and newNode.next.word == phrase[i - 1]:
                    if newNode.prev != None:
                        self.DecreaseValuesOfASubTree (newNode.prev)

                    newNode = newNode.next
                    continue

                else:
                    self.DecreaseValuesOfASubTree (newNode.prev)
                    self.DecreaseValuesOfASubTree (newNode.next)
                    break
            else:
                self.DecreaseValuesOfASubTree (newNode)

        newNode = self.root.next

        for i in range (rootPosition + 1, len(phrase), 1):
            if newNode == None:
                break

            if newNode.word == phrase[i]:
                newNode.mu += 1

                if newNode.next != None and i + 1 < len(phrase) and newNode.next.word == phrase[i + 1]:
                    if newNode.prev != None:
                        self.DecreaseValuesOfASubTree (newNode.prev)

                    newNode = newNode.next
                    continue

                elif newNode.prev != None and i + 1 < len(phrase) and newNode.prev.word == phrase[i + 1]:
                    if newNode.next != None :
                        self.DecreaseValuesOfASubTree (newNode.next)

                    newNode = newNode.prev
                    continue

                else:
                    self.DecreaseValuesOfASubTree (newNode.prev)
                    self.DecreaseValuesOfASubTree (newNode.next)
                    break
            else:
                self.DecreaseValuesOfASubTree (newNode)

    def TreePruning (self, node, k):
        if node is None:
            return

        if node.mu < k:
            node = None
            return

        if node.prev is not None and node.prev.mu < k:
            node.prev = None

        self.TreePruning (node.prev, k)

        if node.next is not None and node.next.mu < k:
            node.next = None

        self.TreePruning (node.next, k)

    def RootToLeafPaths (self, node, nl, ln, rn):
        if node is None:
            return

        nl.append (node)
        self.RootToLeafPaths (node.prev, nl, ln, rn)
        if node.prev is None and node.next is None:
            x = []
            for j in range (0, len(nl), 1):
                x.append(nl[j])
            if len(nl) > 1:
                if nl[0].prev == nl[1]:
                    ln.append(x)
                else:
                    rn.append(x)
            else:
                #If nl contains only one node then it would be appended in the left node list, i.e., ls in this code
                ln.append(x)
                return
        self.RootToLeafPaths (node.next, nl, ln, rn)
        nl.pop()

    def FindPhaseInCandidatePhrases (self, node_list, candidate_phrases):
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

    def FindPhaseExistanceInList (self, node_list, final_node_list):
        ph = self.GetPhrase(node_list)

        exist = False

        for i in range(0, len(final_node_list), 1):
            fph = self.GetPhrase(final_node_list[i])
            if ph == fph:
                exist = True
                break

        return exist

    def FindNodeListToExtractKeyPhrases (self, mu, final_node_list):

        #final_node_list = []
        #print("Tree before pruning: ")
        #self.PrintTree(self.root)

        self.TreePruning (self.root, mu)

        #print("Tree after pruning: ")
        #self.PrintTree(self.root)


        if self.root is None:
            return final_node_list
        elif self.root.mu < mu:
            return final_node_list

        nl = [] #Total node list
        ln = [] #Left node list
        rn = [] #Right node list

        self.RootToLeafPaths(self.root, nl, ln, rn)

        x = []
        x.append(self.root)
        #print("Root: " + self.GetPhrase(x))
        #rabby
        if self.FindPhaseExistanceInList (x, final_node_list) is False:
            final_node_list.append(x)
            #print("ARoot: " + self.GetPhrase(x))

        left_phrase_list = []
        right_phrase_list = []

        #print ("Append from Left: ")
        for i in range(0, len(ln), 1):
            newPhrase = []
            for l in range (0, len(ln[i])):
                newPhrase.insert(0, ln[i][l])
                morePhrase = []
                morePhrase = newPhrase.copy()
                #print("L: " + self.GetPhrase(morePhrase))

                left_phrase_list.append(morePhrase)

            #rabby
                if self.FindPhaseExistanceInList (morePhrase, final_node_list) is False and self.FindPhaseInCandidatePhrases (morePhrase, candidate_phrases)is True:
                    final_node_list.append(morePhrase)
                    #print("AL: " + self.GetPhrase(morePhrase))


        #print ("Append from Right: ")
        for j in range (0, len(rn), 1):
            newPhrase = []
            for m in range (0, len(rn[j]), 1):
                morePhrase = []
                newPhrase.append((rn[j])[m])
                morePhrase = newPhrase.copy()
                #print("R: " + self.GetPhrase(morePhrase))

                right_phrase_list.append(morePhrase)

                if self.FindPhaseExistanceInList (morePhrase, final_node_list) is False and self.FindPhaseInCandidatePhrases (morePhrase, candidate_phrases)is True:
                    final_node_list.append(morePhrase)
                    #print("AR: " + self.GetPhrase(morePhrase))
                    #print(newPhrase.word)

                    #final_node_list.append(newPhrase)

        #
        # Following, we do conccatenation of two sets of strings
        # Details could be found from the following wiki link.
        # https://en.wikipedia.org/wiki/Concatenation
        #
        # Manual concatenation is done since the list contains objects
        #

        #print ("Append from Left and RIght: ")

        for i in range (1, len(left_phrase_list)):
            newPhrase = []
            if len(left_phrase_list[i]) > 1:
                for l in range (0, len(left_phrase_list[i])):
                    newPhrase.append(left_phrase_list[i][l])
            else:
                continue

            #print("L: " + self.GetPhrase(newPhrase))

            for j in range (0, len(right_phrase_list)):
                if len(right_phrase_list[j]) > 1:
                    morePhrase = []
                    morePhrase = newPhrase.copy()
                    for r in range (1, len(right_phrase_list[j])):
                        morePhrase.append(right_phrase_list[j][r])

                    #print("L2R: " + self.GetPhrase(morePhrase))

                    if self.FindPhaseExistanceInList (morePhrase, final_node_list) is False and self.FindPhaseInCandidatePhrases (morePhrase, candidate_phrases)is True:
                        final_node_list.append(morePhrase)
                        #print("AL2R: " + self.GetPhrase(morePhrase))


        '''
        for i in range(0, len(ln), 1):
            for l in range (0, len(ln[i]), 1):
                newPhrase = []
                for m in range (l, len(ln[i]), 1)
                    newPhrase.append(ln[i][m])

            for j in range (0, len(rn), 1):
                newPhrase = []

                for l in range (0, len(ln[i]), 1):
                    morePhrase = []
                    newPhrase.append((ln[i])[l])
                    print("L: " + self.GetPhrase(newPhrase))

                    for m in range (1, len(rn[j]), 1):
                        newPhrase.append((rn[j])[m])
                        morePhrase = newPhrase.copy()
                        print("R: " + self.GetPhrase(morePhrase))

                        if self.FindPhaseExistanceInList (morePhrase, final_node_list) is False:
                            final_node_list.append(morePhrase)
                            print("ALR: " + self.GetPhrase(morePhrase))
        '''

        #print(final_node_list)

        return final_node_list

    #print(final_node_list)

    def GetRoot (self):
        return self.root

    def PrintTree (self, root):
        if root:
            self.PrintTree (root.prev)
            print("....................")
            print(root.word + " " + str(root.mu))
            self.PrintTree (root.next)

    def GetPhrase (self, node_list):
        s = ""
        if len(node_list) > 0:
            for i in range (0, len(node_list)):
                if i != len(node_list) - 1:
                    if node_list[i] is not None:
                        s+=str(node_list[i].word) + " "
                else:
                    if node_list[i] is not None:
                        s+=str(node_list[i].word)
        else:
            print("WARNING: Phrase is empty")

        return s


    def CheckPhraseExist (self, phrase, candidate_phrases):
        ph = self.GetPhrase (phrase)
        if len(ph) > 0:
            for i in range (0, len(candidate_phrases)):
                if ph == candidate_phrases[i]:
                    return True
            return False
        else:
            return False


# In[4]:


class TreeManager:

    kappa = 0

    def FindTFIDFScore(self,word,TFIDF_tuple):
        for i in range(0,len(TFIDF_tuple),1):
            if word == TFIDF_tuple[i][0]:
                return TFIDF_tuple[i][1]
        return 0



    def ProcessCandidatePhrase (self, TFIDF_tuple, candidate_phrases, mu, final_node_list):

        if TFIDF_tuple is None:
            print ("WARNING: ProcessCandidatePhrase():: TFIDF_tuple is None")
            return

        if candidate_phrases is None:
            print ("WARNING: ProcessCandidatePhrase():: candidate_phrases is None")
            return
        #check_repe = []

        #print(TFIDF_tuple)
        #print(candidate_phrases)

        for j in range(0,len(TFIDF_tuple),1):
            word = TFIDF_tuple[j][0]
            score = TFIDF_tuple[j][1]

            tree = Tree()

            tree.AddNode (word, None, 0, 0, score)

            #print(word + " " + tree.root.word + " " + str(tree.root.Root_TFIDF))

            for k in range (0, len(candidate_phrases), 1):
                index = -1
                phrase = candidate_phrases[k].split()
                if tree.root.word in phrase:
                    index = phrase.index(tree.root.word)

                #print (candidate_phrases[k] + " " + tree.root.word + " " + str(index))

                if index == -1:
                    continue
                else:
                    for l in range(index - 1, -1, -1):
                        if tree.AddNode (phrase[l], phrase, l, index, self.FindTFIDFScore (phrase[l], TFIDF_tuple)) is False:
                            break
                    for m in range(index + 1, len(phrase), 1):
                        if tree.AddNode (phrase[m], phrase, m, index, self.FindTFIDFScore (phrase[m], TFIDF_tuple)) is False:
                            break

                    tree.UpdateMuValues (phrase)
                    #tree.TreePruning (tree.GetRoot(), self.kappa)

                #print(candidate_phrases[k] + " -> " + tree.GetRoot().word)
                #tree.PrintTree (tree.GetRoot())

            tree.FindNodeListToExtractKeyPhrases(mu, final_node_list)
            #tree.FindNodeListToExtractKeyPhrases(final_node_list)

    def PrintKeyPhrases (self, final_node_list):

        tuple_list = []

        for i in range (0, len(final_node_list)):
            s = ""
            #print ("-------------------------")
            for j in range (0, len(final_node_list[i])):
                s+=str((final_node_list[i])[j].word)+" "

            tuple_list.append(tuple((s, (final_node_list[i])[j].TFIDF)))
            #print (s)

        tuple_list.sort(key = operator.itemgetter(1), reverse = True)

        print(tuple_list)


    def ReturnKeyPhrases (self, final_node_list, candidate_phrases, text):

        tuple_list = []

        TFIDF_list = []
        mu_list = []

        PD = 0
        for i in range(0,len(candidate_phrases),1):
            if len(candidate_phrases[i].split()) > 0:
                PD += 1

        for i in range(0,len(final_node_list),1):
            x=0
            y=0

            for j in range (0, len(final_node_list[i])):
                x += final_node_list[i][j].TFIDF
                y += final_node_list[i][j].mu
            TFIDF_list.append(x)
            mu_list.append(y)


        #print(max(TFIDF_list))
        #print(min(TFIDF_list))
        text_split = text.split()
        #print(len(text_split))

        for i in range (0, len(final_node_list),1):
            s = ""
            tfidf_score = 0
            mu_score = 0
            ava_TF = 0
            count = 0
            entropy = 0
            root_mu_value = 0
            relevance_score = 0

            for j in range (0, len(final_node_list[i])):
                if final_node_list[i][j].root_status is True:
                    root_mu_value = final_node_list[i][j].mu

            #print ("-------------------------")
            for j in range (0, len(final_node_list[i])):
                s += str((final_node_list[i])[j].word) + " "

                if final_node_list[i][j].TFIDF > 0:
                    entropy += (final_node_list[i][j].TFIDF * (math.log2(1/final_node_list[i][j].TFIDF)))
                else:
                    entropy += 0

                tfidf_score += final_node_list[i][j].TFIDF

                count += 1

                mu_score += final_node_list[i][j].mu

                ava_TF += text_split.count(final_node_list[i][j].word) / len(text_split)

                if root_mu_value > 0:
                    relevance_score +=  final_node_list[i][j].mu / root_mu_value
                else:
                    relevance_score +=  0


            #Ntfidf_score = (((tfidf_score) - min(TFIDF_list))/ (max(TFIDF_list) - min(TFIDF_list)))
            #Nmu_score = (((mu_score) - min(mu_list))/ (max(mu_list) - min(mu_list)))
            #Nlen_score = ((len(s) - 1) / (3 - 1))

            #alpha = 2.3
            #sigma = 3.0
            #B = len(candidate_phrases) / (PD * alpha)
            #B = min(sigma,B)

            #, mu_score/len(text), tfidf_score/len(text)

            #temp_tuple = tuple([s,  (tfidf_score/len(text))*(mu_score/len(text))])
            temp_tuple = tuple([s.strip(),  tfidf_score])


            #print(s + " " + str(relevance_score))
            #temp_tuple = tuple([s, (relevance_score)])

            found = False
            for k in range (0, len(tuple_list)):
                if tuple_list[k][0] == s:
                    found = True
                    break

            if found is False:
                tuple_list.append(temp_tuple)
            #print (s)

        tuple_list.sort(key = operator.itemgetter(1), reverse = True)

        return tuple_list



# In[5]:


class ExtractCandidate:

    def __init__(self, text, select = 1): #creat constructor
        self.text = text

        if select == 1:
            self.extract_candidate_chunks()

    def extract_candidate_chunks(self):
        grammar=r'KT: { (<NN.*>+ <JJ.*>?)|(<JJ.*>? <NN.*>+)}'

        #grammar=r'KT: { (< PRP >? < JJ.∗ > ∗ <NN.∗ > +) }'

        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        # tokenize, POS-tag, and chunk using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(self.text))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
        # join constituent chunk words into a single chunked phrase
        candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(all_chunks, lambda word__pos__chunk: word__pos__chunk[2] != 'O') if key]
        x = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]

        #print(all_chunks)


        data= []

        for i in range(0,len(x),1):
            if len(x[i].split())==1:
                if re.match("^[A-Za-z0-9]*$", x[i]):
                    if len(x[i])>2:
                        data.append(x[i])

            else:
                add=""
                split = x[i].split()
                lenth = len(split)
                for i in range(0,lenth,1):
                    king = re.match("^[A-Za-z0-9]*$", split[i])
                    if len(str(king))>2:
                        add =add +" "+split[i]
                data.append(add.strip())

        return data


    def CleaningCandidatePhrases (self, probable_phrases):
        probpr = []

        for i in range (0, len(probable_phrases), 1):
            words = word_tokenize(probable_phrases[i])
            wa = []
            for w in words:
                x = ps.stem(w)
                wa.append(x)
            probable_phrases[i] = (' '. join(wa))

            #probpr.append(probable_phrases[i])

        for i in range (0, len(probable_phrases), 1):
            for j in range(0,len(probable_phrases[i]),1):
                regex = re.compile('[@_!#$%^&*()<>?/\|}{~:].')
                if regex.search(probable_phrases[i]) == None:
                    probpr.append(probable_phrases[i])
                    break
                else:
                    continue

        for i in range (0, len(probable_phrases), 1):
            temp_probable_phrases = probable_phrases[i].split(" ")
            #print(temp_probable_phrases)
            min_length = True
            for k in range(0,len(temp_probable_phrases),1):
                if len(temp_probable_phrases[k]) < 2:
                    min_length = False


            if min_length:
                count = 0
                for j in range (0, len(probable_phrases), 1):
                    if probable_phrases[i] == probable_phrases[j]:
                        count = count + 1

                    if count > 2:
                        probpr.append(probable_phrases[i])
                        #print(probable_phrases[i])
                        break
            else:
                continue

        return probpr

    def CandidatePhraseHandler (self):
        candidate_phrases = self.extract_candidate_chunks()
        candidate_phrases = self.CleaningCandidatePhrases (candidate_phrases)
        return candidate_phrases


# In[6]:


class NounTFCalculation:

    def NounTF(self,candidate_phrases):

        NounTF_tuple_list = []
        Noun_word = []

        NounTF_chack = []

        for i in range(0,len(candidate_phrases),1):
            word_tokenize = nltk.word_tokenize(candidate_phrases[i])
            tagged_token=nltk.pos_tag(word_tokenize)

            #print(tagged_token)
            for j in range(0,len(tagged_token),1):
                if tagged_token[j][1] == 'NN' or 'NNS' or 'NNP' or 'NNPS':
                    Noun_word.append(tagged_token[j][0])
                    #print()


        #print(Noun_word)

        for i in range(0,len(candidate_phrases),1):
            word_tokenize = nltk.word_tokenize(candidate_phrases[i])
            tagged_token=nltk.pos_tag(word_tokenize)

            #print(tagged_token)
            for j in range(0,len(tagged_token),1):
                if tagged_token[j][1] == 'NN' or 'NNS' or 'NNP' or 'NNPS':
                    #print(tagged_token[j][0]+"..."+str(text_split.count(tagged_token[j][0])))
                    if len(tagged_token[j][0]) > 2:
                        NounTF_chack.append(tagged_token[j][0])
                        #print(tagged_token[j][0]+"....."+str(Noun_word.count(tagged_token[j][0])))

                        #if tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]) / len(Noun_word) )) not in NounTF_tuple_list:
                            #NounTF_tuple_list.append(tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]) / len(Noun_word) )))
                        if Noun_word.count(tagged_token[j][0]) > 10:
                            if tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]))) not in NounTF_tuple_list:
                                NounTF_tuple_list.append(tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]) )))


                        #if tuple((tagged_token[j][0], text_split.count(tagged_token[j][0]) / len(text_split))) not in NounTF_tuple_list:
                         #   NounTF_tuple_list.append(tuple((tagged_token[j][0], text_split.count(tagged_token[j][0])/ len(text_split))))
                    #if tagged_token[j][0] not in
        #for k in range(0,len(NounTF_chack),1):
         #   if NounTF_chack.count(NounTF_chack[k]) > 1:
          #      if tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]))) not in NounTF_tuple_list:
           #         NounTF_tuple_list.append(tuple((tagged_token[j][0], Noun_word.count(tagged_token[j][0]) )))

        #print(NounTF_tuple_list)
        return NounTF_tuple_list


# In[7]:


class ORPNode:
    def __init__(self, papername, similarityscore):
        self.PaperName = papername
        self.SimilarityScore = similarityscore


# In[8]:


class CosineSimilarityWithMU:

    def Sorting(self,reference_cosine_list ):
        for i in range(0,len(reference_cosine_list),1):
            for j in range(len(reference_cosine_list)-1-i):
                if reference_cosine_list[j].SimilarityScore < reference_cosine_list[j+1].SimilarityScore:
                    reference_cosine_list[j], reference_cosine_list[j+1] = reference_cosine_list[j+1], reference_cosine_list[j]


    def Root(self,Keyphrases):

        return Keyphrases

    #def cosine_similarity(self,vector1,vector2):
     #   dot_product = np.dot(vector1, vector2)
      #  norm_a = np.linalg.norm(vector1)
       # norm_b = np.linalg.norm(vector2)
        #return dot_product / (norm_a * norm_b)

    def BestCosineSimilarityWithMU(self,paper_name,Keyphrases,root_keyphrase_list,reference_cosine_list):


        list_1 = []
        list_2 = []
        Dot_product = 0
        sum_sqr_1=0
        sum_sqr_2 = 0
        cosine_similarity = 0

        #sum_root = 0
        #sum_keyphrases = 0

        #for i in range(0,len(root_keyphrase_list),1):
         #   sum_root += root_keyphrase_list[i][1]

        #for i in range(0,len(Keyphrases),1):
         #   sum_keyphrases += Keyphrases[i][1]

        #print("Len of the root: "+str(len(root_keyphrase_list)))
        #print("sum_root: "+str(sum_root))
        #print("sum_keyphrases: "+str(sum_keyphrases))

        for i in range(0,len(root_keyphrase_list),1):
            list_1.append(root_keyphrase_list[i][1]  )
            list_2.append(0)


        for i in range(0,len(root_keyphrase_list),1):
            for j in range(0,len(Keyphrases),1):
                if root_keyphrase_list[i][0].strip() == Keyphrases[j][0].strip():
                    #print(i)
                    list_2[i] = Keyphrases[j][1]

        #print(list_1)
        #print(list_2)

        #cos_sim = self.cosine_similarity(list_1,list_2)

        #print(cos_sim)
        #print(type(cos_sim.tolist()))

        #reference_cosine_list.append(ORPNode(paper_name, cos_sim.tolist()))

        #return reference_cosine_list


        for j in range(0,len(list_1),1):
            sum_sqr_1 +=  (list_1[j] * list_1[j])


        ##sum_sqr_1 += sum_sqr_1 + (list_1[i]*list_1[i] for i in range(list_1))


        for j in range(0,len(list_2),1):
            #list_2.append(Keyphrases[j][1])
            sum_sqr_2 +=  (list_2[j] * list_2[j])

        #Dot_product = numpy.dot(list_1,list_2)
        #if len(list_1) == len(list_2):

        #Dot_product = sum([x*y for x,y in zip(list_1,list_2)])

        Dot_product = np.dot(list_1, list_2)
        #norm_a = np.linalg.norm(list_1)
        #norm_b = np.linalg.norm(list_2)

        #print(list_2)

        norm_a = math.sqrt(sum_sqr_1)
        norm_b = math.sqrt(sum_sqr_2)


        #print("Dot_product: "+str(Dot_product))
        #print("sum_sqr_1: "+str(norm_a))
        #print("sum_sqr_2: "+str(norm_b))
        #print("norm_a*norm_b: "+str((norm_a * norm_b)))

        if (norm_a*norm_b) != 0:
            cosine_similarity = (Dot_product/ (norm_a*norm_b)) # / (norm_a*norm_b)
        else:
            cosine_similarity = 0

        #print(cosine_similarity)
        #print("\n")

        reference_cosine_list.append(ORPNode(paper_name, cosine_similarity))

        return reference_cosine_list





# In[9]:


class JaccardSimilarity:


    def Sorting(self,reference_jaccard_list ):
        for i in range(0,len(reference_jaccard_list),1):
            for j in range(len(reference_jaccard_list)-1-i):
                if reference_jaccard_list[j].SimilarityScore < reference_jaccard_list[j+1].SimilarityScore:
                    reference_jaccard_list[j], reference_jaccard_list[j+1] = reference_jaccard_list[j+1], reference_jaccard_list[j]




    def Root(self,Keyphrases):
        list_1 = []
        for j in range(0,len(Keyphrases),1):
            list_1.append(Keyphrases[j][0])
        return list_1


    def BestJaccardSimilarity(self,paper_name,Keyphrases,list_1,reference_jaccard_list):

        #list_2 = []

        #list_1 = list_1
        #list_2 = Keyphrases

        #list_3 = list_1+list_2

        #print("len of list_1: "+str(len(list_1)))
        #print("len of list_2: "+str(len(list_2)))
        #print("len of list_3: "+str(len(list_3)))

        #Jaccard_similarity = float(len(list_3)) / (len(list_1) + len(list_2) - len(list_3))

        #reference_jaccard_list.append(ORPNode(paper_name, Jaccard_similarity))

        #return reference_jaccard_list

        #print("list: "+str(list_1))
        #print("Keyphrases: "+str(Keyphrases))

        count = 0

        #.strip()

        for i in range(0,len(list_1),1):
            #print(list_1[i].strip())
            #print(Keyphrases[i][0].strip())
            for j in range(0,len(Keyphrases),1):
                #print()
                if list_1[i].strip() == Keyphrases[j][0].strip():

                    #print(list_1[i].strip())

                    count += 1

        #print(count)

        Jaccard_similarity = float(count / (len(list_1)+len(Keyphrases)-count))
        reference_jaccard_list.append(ORPNode(paper_name, Jaccard_similarity))
        ##intersection = len(list(set(list1).intersection(list2)))
        #print(list(set(list1).intersection(list2)))
        ##union = (len(list1) + len(list2)) - intersection
        #return float(intersection / union)
        ##Jaccard_similarity = float(intersection / union)

        ##reference_jaccard_list.append(ORPNode(paper_name, Jaccard_similarity))

        return reference_jaccard_list




# In[ ]:


treemanager = TreeManager()
ntf = NounTFCalculation()
cs = CosineSimilarityWithMU()
js = JaccardSimilarity()

mu = 2
Top_n = 15


orp_path = []
reference_cosine_list = []
reference_jaccard_list = []

#,encoding="utf-8"

#KeyPhrase_write = open("small_test_data/AAMAS/All keyphrase with our algo.txt", "w",encoding="ISO-8859-1")

#....off by rabby.....#
#KeyPhrase_count_in_a_paper = open("small_test_data/AAMAS/test1.txt", "r",encoding="utf-8")
#KeyPhrase_count_in_a_paper_read = KeyPhrase_count_in_a_paper.read()
#....off by rabby.....#

#print(KeyPhrase_count_in_a_paper_read.split("."))

#Keyphrases_pro_type = "combined"

#if Keyphrases_pro_type == "combined":
# Gold-standard Keyphrases start

# /S10-1041.pdf.txt

root_paper = open("ORP/Cosine Similarity/Base_txt/acl17.txt", "r",encoding="utf-8")
root_paper_read = root_paper.read()

ps_stem_list = []
split = root_paper_read.split()
for j in range(0,len(split),1):
    ps_stem_list.append(ps.stem(split[j].lower()))
text = ' '.join(ps_stem_list)
text_lenth = len(text.split(" "))
final_node_list = []
extract_candidate = ExtractCandidate(text)
candidate_phrases = extract_candidate.CandidatePhraseHandler()
candidate_phrases_count_3 = []
for x in range(0,len(candidate_phrases),1):
    if candidate_phrases.count(candidate_phrases[x]) > 2 :
        candidate_phrases_count_3.append(candidate_phrases[x])
NounTF_tuple = ntf.NounTF(candidate_phrases_count_3)
treemanager.ProcessCandidatePhrase (NounTF_tuple, candidate_phrases_count_3, mu, final_node_list)
Keyphrases = treemanager.ReturnKeyPhrases (final_node_list, candidate_phrases_count_3,text)
root_score_list = cs.Root(Keyphrases)
root_keyphrase_list = js.Root(Keyphrases)

#print(root_keyphrase_list)

#print(cs.Root(Keyphrases))

#orp_path

#print("root: ")
#print("\n")
#aaaa = []
#for zz in range(0,len(Keyphrases),1):
 #   aaaa.append(Keyphrases[zz][0])



#print(aaaa)



#

#/0212020/domain-specific keyphrase extraction
full_path = [os.path.join(r,file) for r,d,f in os.walk("ORP/Cosine Similarity/New folder/LVL-2") for file in f]


for i in range(0,len(full_path),1):
    # KeyPhrase_count_in_a_paper_read(TFIDF) end
    print("..................")
    print(i)
    #print(full_path_KeyPhrase_count_in_a_paper[i])
    print(full_path[i])
    print("..................")

    #Full text file read start

    with open(full_path[i], "r",encoding="utf-8") as f:

        ps_stem_list = []

        text_read = f.read()
        split = text_read.split()
        for j in range(0,len(split),1):
            ps_stem_list.append(ps.stem(split[j].lower()))
        text = ' '.join(ps_stem_list)

        text_lenth = len(text.split(" "))



        final_node_list = []

        extract_candidate = ExtractCandidate(text)
        candidate_phrases = extract_candidate.CandidatePhraseHandler()


        candidate_phrases_count_3 = []
        for x in range(0,len(candidate_phrases),1):
            if candidate_phrases.count(candidate_phrases[x]) > 2 :
                candidate_phrases_count_3.append(candidate_phrases[x])

        #print(candidate_phrases_count_3)
        #break


        NounTF_tuple = ntf.NounTF(candidate_phrases_count_3)

        #print(NounTF_tuple)
        #break

        treemanager.ProcessCandidatePhrase (NounTF_tuple, candidate_phrases_count_3, mu, final_node_list)

        #treemanager.PrintKeyPhrases (final_node_list)
        #print("candidate_phrases: ")
        #print(candidate_phrases)
        #print("........................")

        #break



        Keyphrases = treemanager.ReturnKeyPhrases (final_node_list, candidate_phrases_count_3,text)

        #print("Keyphrase: ")
        #print(Keyphrases)
        #print("...............................")

        #abc=[]

        #for aa in range(0,len(Keyphrases),1):
         #   abc.append(Keyphrases[aa][0])
            #print(Keyphrases[aa][0])
        #print(abc)


        #print(Keyphrases)

        path_split = full_path[i].split("\\")
        paper_name = ''.join(path_split[1])
        #paper_name = filename_str.split(".")

        #print(paper_name)




        #if len(Keyphrases)>19:
        cs.BestCosineSimilarityWithMU(paper_name,Keyphrases,root_score_list,reference_cosine_list)

        #print(cs.BestCosineSimilarityWithMU(paper_name,Keyphrases,cs.Root(Keyphrases)))

        js.BestJaccardSimilarity(paper_name,Keyphrases,root_keyphrase_list,reference_jaccard_list)

cs.Sorting(reference_cosine_list)
js.Sorting(reference_jaccard_list)

#print(reference_cosine_list)
print("\n")
for j in range(0,len(reference_cosine_list),1):

    #print(reference_cosine_list[j].SimilarityScore)

    if reference_cosine_list[j].SimilarityScore >0.10:
        print("Paper Name: "+str(reference_cosine_list[j].PaperName))
        print("Cosine Similarity Score: "+str(reference_cosine_list[j].SimilarityScore))
        print("\n")
    #else:
     #   break


#print("Paper Name: "+str(reference_cosine_list[0].PaperName))
#print("Cosine Similarity Score: "+str(reference_cosine_list[0].SimilarityScore))

#for j in range(0,len(reference_jaccard_list),1):
#    if reference_jaccard_list[j].SimilarityScore >0.10:
#        print("Paper Name: "+str(reference_jaccard_list[j].PaperName))
#        print("Jaccard Similarity Score: "+str(reference_jaccard_list[j].SimilarityScore))
#        print("\n")
#    else:
#        break

#print("Paper Name: "+str(reference_jaccard_list[0].PaperName))
#print("Jaccard Similarity Score: "+str(reference_jaccard_list[0].SimilarityScore))



        #if i == 1:
         #   break

        #break



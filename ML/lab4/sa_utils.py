import enum
import numpy as np
import pandas as pd
import re
import nltk
import csv


# import os
# import system


def pause():
    programPause = input("Paused. Press the <ENTER> key to continue...")


def readFile(filename=None):
    # READFILE reads a file and returns its entire contents
    #   file_contents = READFILE(filename) reads a file and returns its entire
    #   contents in file_contents

    with open(filename, "r") as file:
        file_contents = file.read().replace("\n", "")
    return file_contents


def getVocabList():
    #   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    #   and returns a cell array of the words in vocabList.
    
    with open('vocab.txt') as f:
        reader = csv.reader(f,  delimiter='\t')
        vocabList = [word[1] for word in reader]

    return vocabList


def processEmail(email_contents=None):
    #   word_indices = PROCESSEMAIL(email_contents) preprocesses
    #   the body of an email and returns a list of indices of the
    #   words contained in the email.

    # Load Vocabulary
    vocabList = getVocabList()
    # Init return value
    word_indices = np.array([])
    # ========================== Preprocess Email ===========================

    # Headers
    # Handle them bellow  if you are working with raw emails with the
    # full headers

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # it with a space
    pattern = "<[^<]+?>"
    email_contents = re.sub(pattern, " ", email_contents)

    # Look for one or more characters between 0-9
    pattern = r"[0-9]+"
    # Match all digits in the string and replace them with 'number'
    email_contents = re.sub(pattern, " number", email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    pattern = r"(http|https)\S+"
    email_contents = re.sub(pattern, "httpaddr", email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    pattern = r"[\w.+-]+@[\w-]+\.[\w.-]+"
    email_contents = re.sub(pattern, "emailaddr", email_contents)

    pattern = r"\$"
    email_contents = re.sub(pattern, "dollar", email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    #print("\n==== Processed Email ====\n\n" % ())
    # Process file
    l = 0
    for s in re.split("[ ;\-,]", email_contents):
        # Tokenize and also get rid of any punctuation
        s = re.sub(r"[^\w\s]", "", s)
        # Remove any non alphanumeric characters
        s = re.sub("[^0-9a-zA-Z]+", " ", s)

        # Stem the word
        ps = nltk.stem.PorterStemmer()
        s = ps.stem(s)
        # Skip the word if it is too short
        if len(s) < 1:
            continue
        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices
        #               vector (e.g., word_indices = [word_indices ; 18]; ).
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        for i in np.arange(0, len(vocabList)):
            #  if (str(vocabList[i]) == str(s)):
            if vocabList[i] == s:
                word_indices = np.append(word_indices, i)

        # =============================================================
        # Print to screen, ensuring that the output lines are not too long
        '''if (l + len(s) + 1) > 78:
            print('\n' % ())
            l = 0
        print('%s ' % (s))
        l = l + len(s) + 1'''

    # Print footer
    # print("\n\n=========================\n" % ())
    return word_indices


def emailFeatures(word_indices=None):
    #   x = EMAILFEATURES(word_indices) takes in a word_indices vector and
    #   produces a feature vector from the word indices.
    vocabList = getVocabList()
    # Total number of words in the dictionary
    n = len(vocabList)

    # You need to return the following variables correctly.
    x = np.zeros(n, int)
    # ====================== YOUR CODE HERE ======================

    # Instructions: Fill in this function to return a feature vector for the
    #               given email (word_indices). To help make it easier to
    #               process the emails, we have have already pre-processed each
    #               email and converted each word in the email into an index in
    #               a fixed dictionary (of 1899 words). The variable
    #               word_indices contains the list of indices of the words
    #               which occur in one email.

    #               Concretely, if an email has the text:
    #                  The quick brown fox jumped over the lazy dog.
    #               Then, the word_indices vector for this text might look
    #               like:
    #                   60  100   33   44   10     53  60  58   5
    #               where, we have mapped each word onto a number, for example:
    #                   the   -- 60
    #                   quick -- 100
    #                   ...
    #              (note: the above numbers are just an example and are not the
    #               actual mappings).

    #              Your task is take one such word_indices vector and construct
    #              a binary feature vector that indicates whether a particular
    #              word occurs in the email. That is, x(i) = 1 when word i
    #              is present in the email. Concretely, if the word 'the' (say,
    #              index 60) appears in the email, then x(60) = 1. The feature
    #              vector should look like:

    #              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];

    for word_index in word_indices:
        x[int(word_index)] = 1
    return x

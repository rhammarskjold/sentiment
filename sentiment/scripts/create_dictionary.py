""" create_dictionary takes the name of a vocab file and processes it returning both a
    dictionary mapping words to indecies and an array that holds the word for each index"""
def create_dictionary(vocabFileName):
    vocabFile = open(vocabFileName, "r")
    wordToId = dict()
    idToWord = []
    counter = 0
    for word in vocabFile.read().splitlines():
        idToWord.append(word)
        wordToId[word] = counter
        counter += 1

    return (wordToId, idToWord)

# Attempts to answer the question through somewhat educated random guessing
# Tokenizes text, picks a starting and ending point, and returns a well-formed response.
# Rejects questions at random.
# Used to test eval.py.
from random import random, sample, randrange
from functools import reduce
from nltk.tokenize.stanford import StanfordTokenizer

tokenizer = StanfordTokenizer(options={
    'ptb3Escaping': False
})

# (impossible (bool), starting index, ending index)
def eval(context, question):
    # Randomly reject 20% of questions as not being in the text:
    if random() < 0.2:
        return (True, 0, 0)

    # Note this adds nothing for spacing:
    textToTokenizedLength = lambda total, word: total + len(word)
    tokenized = tokenizer.tokenize(context)

    startingWordIndex = randrange(0, len(tokenized) - 2)
    word = tokenized[startingWordIndex]
    # TODO: This creates errors by not perfectly undoing the original tokenizing:
    originalTextIndex = reduce(textToTokenizedLength, tokenized[0:startingWordIndex], 0)
    # TODO: So backtrack a bit from the overzealous estimate, and find the word:
    startingTextIndex = context.index(word, originalTextIndex)

    # Select up to 5 more tokens:
    endingWordIndex = randrange(startingWordIndex + 1, min(len(tokenized) - 1, startingWordIndex + 4))
    word = tokenized[endingWordIndex]
    originalTextIndex = reduce(textToTokenizedLength, tokenized[0:endingWordIndex], 0)
    endingTextIndex = context.index(word, originalTextIndex)

    return (False, startingTextIndex, endingTextIndex)


# Attempts to answer the question through somewhat educated random guessing
# Tokenizes text, picks a starting and ending point, and returns a well-formed response.
# Rejects questions at random.
# Used to test eval.py.
from random import random, sample, randrange
from functools import reduce
import os
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser

parser = None
config = {
    'isManagingServer': False
}
STANFORD = os.path.join("models", "stanford-corenlp-full-2018-10-05")
server = CoreNLPServer(
   os.path.join(STANFORD, "stanford-corenlp-3.9.2.jar"),
   os.path.join(STANFORD, "stanford-corenlp-3.9.2-models.jar"),
)

def setup(manageServerInternally):
    config['isManagingServer'] = manageServerInternally

    if manageServerInternally:
        print("Starting CoreNLP server...")
        server.start()

# (impossible (bool), starting index, ending index)
def eval(context, question):
    # Randomly reject 20% of questions as not being in the text:
    if random() < 0.2:
        return (True, 0, 0)

    # Note this adds nothing for spacing:
    textToTokenizedLength = lambda total, word: total + len(word)
    tokenized = list(parser.tokenize(context))

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

# Teardown code:
def stop():
    if config['isManagingServer']:
        server.stop()

parser = CoreNLPParser(url='http://localhost:9000')

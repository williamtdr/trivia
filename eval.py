# Network evaluation script

import json
from ie_solution import eval, setup, stop
import multiprocessing as mp
import sys

USE_PRODUCTION_DATA = False
MANAGE_CORENLP_INTERNALLY = False

totalQuestions = 0
totalPassages = 0
answeredQuestions = 0

def findBaselineStats(topics):
    global totalQuestions, totalPassages

    for topic in topics:
        for passage in topic['paragraphs']:
            totalPassages += 1
            totalQuestions += len(passage['qas'])

def evaluatePassage(x):
    passage, answeredPassages, answeredQuestions, totalQuestions, totalPassages = x
    context = passage['context']
    questions = passage['qas']
    rightPassage = 0
    wrongPassage = 0
    allGlobalStats = []

    print(
        f'>>> processing... {answeredQuestions} of {totalQuestions} Qs, {answeredPassages} of {totalPassages} passages')
    print('***')
    print(context)
    print('***')

    def right(answer=None):
        if answer is None:
            print("right! model predicted question is impossible, and it is.")
        else:
            print(f"right! predicted: {answer}")

    def wrong(answer, isActuallyImpossible, evalThinksImpossible, correctAnswer = None):
        if isActuallyImpossible and not evalThinksImpossible:
            return print(f">>> wrong. question is impossible. predicted: {answer}")

        if answer == '' and not isActuallyImpossible:
            return print(">>> wrong. question is fine, model predicted as impossible.")

        if correctAnswer is not None:
            return print(f">>> wrong. predicted: {answer}, correct: {correctAnswer}")
        else:
            return print(f">>> wrong. predicted: {answer}")

    for questionObj in questions:
        question = questionObj['question']
        isImpossible = questionObj['is_impossible']
        trainingAnswers = None
        if 'answers' in questionObj:
            trainingAnswers = questionObj['answers']

        # print(question)

        answers = []
        if trainingAnswers:
            for potentialSolutionObj in trainingAnswers:
                answers.append(potentialSolutionObj['text'])

        answers = list(set(answers))
        evalIsImpossible, evalStartIndex, evalEndIndex, evalGlobalStats = eval(context, question, answers)
        allGlobalStats += [ evalGlobalStats ]
        evalAnswer = context[evalStartIndex:evalEndIndex].strip()
        lastTrainingAnswer = None

        for potentialSolutionObj in trainingAnswers:
            text = potentialSolutionObj['text']
            lastTrainingAnswer = text

        if evalIsImpossible and isImpossible:
            rightPassage += 1
            right()
        elif evalIsImpossible != isImpossible:
            wrongPassage += 1
            wrong(evalAnswer, isImpossible, evalIsImpossible, lastTrainingAnswer)
        else:
            foundCorrectAnswer = False
            for potentialSolutionObj in trainingAnswers:
                start = potentialSolutionObj['answer_start']
                text = potentialSolutionObj['text']

                if foundCorrectAnswer:
                    continue

                # start == evalStartIndex?
                if text.strip().lower() == evalAnswer.lower():
                    foundCorrectAnswer = True
                    rightPassage += 1
                    right(evalAnswer)

            if not foundCorrectAnswer:
                wrongPassage += 1
                wrong(evalAnswer, isImpossible, evalIsImpossible, lastTrainingAnswer)

    print(f'>>> for passage, right={rightPassage} wrong={wrongPassage}')
    return (rightPassage, wrongPassage, allGlobalStats)

def evaluate(topics):
    global totalQuestions, totalPassages, pool

    rightTotal = 0
    wrongTotal = 0
    answeredPassages = 0
    answeredQuestions = 0
    # global stats:
    numPossible = 0
    numPossibleWithRelevantSentences = 0
    numIntentionallyImpossible = 0
    numImpossible = 0

    for topic in topics:
        title = topic['title']
        rightTopic = 0
        wrongTopic = 0

        print(f">>> Start topic {title}!")
        passages = topic['paragraphs']
        passagesWithArguments = map(lambda x: (x, answeredPassages, answeredQuestions, totalQuestions, totalPassages), passages)

        newCorrectnessValues = list(pool.map(evaluatePassage, passagesWithArguments))
        answeredPassages += len(passages)
        print(f">>> Finished topic {title}.")

        for passageResult in newCorrectnessValues:
            rightTopic += passageResult[0]
            wrongTopic += passageResult[1]
            rightTotal += passageResult[0]
            wrongTotal += passageResult[1]
            answeredQuestions += passageResult[0] + passageResult[1]

            for question in passageResult[2]:
                numPossible += question['numPossible']
                numPossibleWithRelevantSentences += question['numPossibleWithRelevantSentences']
                numIntentionallyImpossible += question['numIntentionallyImpossible']
                numImpossible += question['numImpossible']

        print(f">>> right={rightTotal}, wrong={wrongTotal}, right for topic={rightTopic}, wrong for topic={wrongTopic}")
        print(f">>> possible with noun phrases={numPossible}, "
              f"possible with sentence context={numPossibleWithRelevantSentences}, "
              f"intentionally impossible={numIntentionallyImpossible}, "
              f"impossible by current standards={numImpossible}, "
              f"total={numImpossible + numPossible + numPossibleWithRelevantSentences + numIntentionallyImpossible}")

    return rightTotal, wrongTotal

if __name__ == '__main__':
    fileName = "squad-train-v2.0.json" if USE_PRODUCTION_DATA else "squad-dev-v2.0.json"
    print(f"Loading SQuAD Dataset ({'Training' if USE_PRODUCTION_DATA else 'Dev'})...")

    with open('data/' + fileName) as f:
        data = json.load(f)

    print("Preparing model...")
    setup(MANAGE_CORENLP_INTERNALLY)

    cmdLineArgs = sys.argv[1:]
    concurrency = mp.cpu_count()
    if len(cmdLineArgs) >= 1:
        concurrency = int(cmdLineArgs[0])
        print(f"Starting {concurrency} threads based on command line argument.")
    else:
        print(f"Starting {concurrency} threads based on number of processors.")

    pool = mp.Pool(concurrency)
    findBaselineStats(data['data'])
    right, wrong = evaluate(data['data'])

    print(f"Evaluation complete. Correct: {str(right)}/{str(right + wrong)}")
    print("Stopping dependent systems...")
    stop()
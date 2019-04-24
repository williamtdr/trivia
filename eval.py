# Network evaluation script

import json
from ml_type_solution import eval, setup, stop

USE_PRODUCTION_DATA = False
MANAGE_CORENLP_INTERNALLY = False

totalQuestions = 0
totalPassages = 0
answeredQuestions = 0
answeredPassages = 0

def findBaselineStats(topics):
    global totalQuestions, totalPassages
    for topic in topics:
        for passage in topic['paragraphs']:
            totalPassages += 1

            for q in passage['qas']:
                totalQuestions += 1

def evaluate(topics):
    global answeredPassages, answeredQuestions

    rightTotal = 0
    wrongTotal = 0

    for topic in topics:
        title = topic['title']
        rightTopic = 0
        wrongTopic = 0

        print("Start topic " + title + "!")
        for passage in topic['paragraphs']:
            context = passage['context']
            questions = passage['qas']
            rightPassage = 0
            wrongPassage = 0
            answeredPassages += 1

            print('>>> processing... {0} of {1} Qs, {2} of {3} passages. {4} correct, {5} wrong.'.format(answeredQuestions, totalQuestions, answeredPassages, totalPassages, rightTotal, wrongTotal))
            print('***')
            print(context)
            print('***')

            def right(answer):
                print("right! predicted: " + answer)

            def wrong(answer, isActuallyImpossible, evalThinksImpossible, correctAnswer = None):
                if isActuallyImpossible and not evalThinksImpossible:
                    return print("wrong. question is impossible. predicted: " + answer)

                if answer == '' and not isActuallyImpossible:
                    return print("wrong. question is fine.")

                if correctAnswer != None:
                    return print("wrong. predicted: " + answer + ", correct: " + correctAnswer)
                else:
                    return print("wrong. predicted: " + answer)

            for questionObj in questions:
                question = questionObj['question']
                isImpossible = questionObj['is_impossible']
                trainingAnswers = None
                if 'answers' in questionObj:
                    trainingAnswers = questionObj['answers']

                print(question)
                answeredQuestions += 1

                evalIsImpossible, evalStartIndex, evalEndIndex = eval(context, question)
                evalAnswer = context[evalStartIndex:evalEndIndex].strip()
                lastTrainingAnswer = None

                if evalIsImpossible != isImpossible:
                    wrongPassage += 1
                    wrongTopic += 1
                    wrongTotal += 1
                    wrong(evalAnswer, isImpossible, evalIsImpossible)
                else:
                    foundCorrectAnswer = False
                    for potentialSolutionObj in trainingAnswers:
                        start = potentialSolutionObj['answer_start']
                        text = potentialSolutionObj['text']
                        lastTrainingAnswer = text

                        if foundCorrectAnswer:
                            continue

                        # start == evalStartIndex?
                        if text.strip() == evalAnswer:
                            foundCorrectAnswer = True
                            rightPassage += 1
                            rightTopic += 1
                            rightTotal += 1
                            right(evalAnswer)

                    if not foundCorrectAnswer:
                        wrongPassage += 1
                        wrongTopic += 1
                        wrongTotal += 1
                        wrong(evalAnswer, isImpossible, evalIsImpossible, lastTrainingAnswer)

    return (rightTotal, wrongTotal)

fileName = "squad-train-v2.0.json" if USE_PRODUCTION_DATA else "squad-dev-v2.0.json"
print("Loading SQuAD Dataset (" + ("Training" if USE_PRODUCTION_DATA else "Dev") + ")...")

with open('data/' + fileName) as f:
    data = json.load(f)

print("Preparing model...")
setup(MANAGE_CORENLP_INTERNALLY)

findBaselineStats(data['data'])
right, wrong = evaluate(data['data'])
print("Evaluation complete. Correct: " + str(right) + "/" + str(right + wrong))

print("Stopping dependent systems...")
stop()
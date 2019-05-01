import sys
import inquirer
from ie_solution import eval, setup, stop
from termcolor import colored

# ask to load in the context
# options 1. ask question 2. load new context 3. quit the program

contentInput = ""

def content():
    global contentInput

    content = [
      inquirer.Text('context', message="Please input your context")]
    articles = inquirer.prompt(content)
    if articles == None:
        exit(0)

    articles = articles['context']
    contentInput = articles

    return articles

def option():
    options = [
        inquirer.List('options',
                message="Choose one option",
                choices=['Ask a question', 'Load new context', 'Add context', 'Quit'],
            ),
    ]
    optionchosen = inquirer.prompt(options)
    return optionchosen

def prompt():
    global contentInput
    opt = option()

    if opt == None:
        exit(0)

    if opt['options'] == "Ask a question":
        Q = [inquirer.Text('question', message="Please type your question")]
        question = inquirer.prompt(Q)['question']
        answer = eval(contentInput, question)
        if not answer[0]:
            start, end = answer[1], answer[2]
            answer = contentInput[start:end].strip()

            print("***")
            print(
                colored(contentInput[0:start], 'yellow'),
                colored(contentInput[start:end], 'magenta'),
                colored(contentInput[end:], 'yellow'),
                sep=''
            )
            print("***")
            print(f">>> Model predicted: {answer}")
        else:
            print(">>> Model predicted: Question is impossible.")

        prompt()
    elif opt['options'] == "Load new context":
        contentInput = ""
        content()
        prompt()
    elif opt['options'] == "Add context":
        oldcontent = contentInput
        content()
        contentInput = oldcontent + contentInput
        prompt()
    elif opt['options'] == "Quit":
        print("Goodbye!")
        stop()
        exit(0)

        return

setup(False)
print("Welcome!")
content()
prompt()

#print(question)

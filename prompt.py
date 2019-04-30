import sys
import inquirer
from ie_solution import eval, setup, stop

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
            # todo: highlight the answer in the original text
            print(f">>> Model predicted: {answer[2]} starting at index {answer[1]} in text.")
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
        exit(0)

        return

print("Welcome!")
content()
prompt()

#print(question)

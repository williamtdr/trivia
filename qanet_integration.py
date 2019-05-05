import requests

endpoint = "http://127.0.0.1:8080"

def getAnswer(passage, question):
    return requests.post(endpoint + "/answer", {
        'passage': passage,
        'question': question
    })['answer']

def setupQANet():
    try:
        print("Checking connection to tensorflow QANet network...")

        requests.get(endpoint)
    except BaseException:
        print("Error connecting to ML instance! Make sure the server is running in the background.")
        print("The relevant command can be found in the README.")

        exit(1)
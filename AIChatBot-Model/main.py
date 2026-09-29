from models.model import main

print("Welcome to LLM Service:")
print("Enter 'break' to exit")

while True:

    prompt = input("Enter your prompt to instruct agent: ")

    if prompt.lower() == "break":
        print("Exiting...")
        break

    response = main(prompt)

    print("\nModel response:")
    print(response)
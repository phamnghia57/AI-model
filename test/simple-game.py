import random

while True:
    choices = ["Rock", "Paper", "Scissors"]
    comp = random.choice(choices)
    user = input("Rock / Paper / Scissors (or 'Quit' to exit): ").capitalize()

    if user == "Quit":
        print("Thank for playing")
        break

    if user not in choices:
        print("Invalid choic, Try again.")
    
    print(f"Computer choice: {comp}")

    if user == comp:
        print("It's a tie")
    elif (user == "Rock" and comp == "Scissors") or \
         (user == "Paper" and comp == "Rock") or \
         (user == "Scissors" and comp == "Paper"):
        print("You win")
    else:
        print("Computer win")

    print("-" * 20)
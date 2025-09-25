def check_length(text="Too short"):
    if len(text) < 10:
        print("Too short")
    else:
        print(text)

def main():
    while True:
        user_text = input("Write something (quit ends): ")
        if user_text == "quit":
            break
        check_length(user_text)

if __name__ == "__main__":
    main()

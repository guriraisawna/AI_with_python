def main():
    cart = []

    while True:
        action = input("Would you like to \n(1)Add or \n(2)Remove items or \n(3)Quit?: ")

        if action == "1":
            new_item = input("What will be added?: ")
            cart.append(new_item)

        elif action == "2":
            if len(cart) == 0:
                print("The list is empty, nothing to remove.")
                continue

            print(f"There are {len(cart)} items in the list.")
            try:
                item_index = int(input("Which item is deleted?: "))
                if 0 <= item_index < len(cart):
                    cart.pop(item_index)
                else:
                    print("Incorrect selection.")
            except ValueError:
                print("Incorrect selection.")

        elif action == "3":
            print("The following items remain in the list:")
            for item in cart:
                print(item)
            break

        else:
            print("Incorrect selection.")


main()

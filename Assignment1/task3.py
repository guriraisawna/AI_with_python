def main():
    product_prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
    total_amount = 0

    print("Supermarket")
    print("===========")

    while True:
        try:
            product_choice = int(input("Please select product (1-10) 0 to Quit: "))
        except ValueError:
            print("Invalid input, please enter a number.")
            continue

        if product_choice == 0:
            break
        elif 1 <= product_choice <= 10:
            selected_price = product_prices[product_choice - 1]
            total_amount += selected_price
            print(f"Product: {product_choice} Price: {selected_price}")
        else:
            print("Invalid product number. Please choose between 1 and 10.")

    print(f"Total: {total_amount}")

    while True:
        try:
            payment_amount = int(input("Payment: "))
            if payment_amount < total_amount:
                print("Not enough money, please enter again.")
                continue
            break
        except ValueError:
            print("Invalid input, please enter a number.")

    change_amount = payment_amount - total_amount
    print(f"Change: {change_amount}")


if __name__ == "__main__":
    main()

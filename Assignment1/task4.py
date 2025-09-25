def my_split(text, delimiter):
    split_items = []
    current_item = ""
    for char in text:
        if char == delimiter:
            split_items.append(current_item)
            current_item = ""
        else:
            current_item += char
    split_items.append(current_item)
    return split_items

def my_join(items, delimiter):
    joined_text = ""
    for i in range(len(items)):
        joined_text += items[i]
        if i < len(items) - 1:
            joined_text += delimiter
    return joined_text

def main():
    user_input = input("Please enter sentence: ")
    words = my_split(user_input, " ")
    joined_words = my_join(words, ",")
    print(joined_words)
    for word in words:
        print(word)

if __name__ == "__main__":
    main()

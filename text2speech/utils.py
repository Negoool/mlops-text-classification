from num2words import num2words


def convert_to_words(text):
    words_list = [num2words(tok) if tok.isdigit() else tok for tok in text.split(" ")]
    return " ".join(words_list)

START = "<s>"
END = "</s>"


def read_data(filename):
    with open(filename, "r") as file:
        all_lines = file.readlines()

    return_lines = []
    for line in all_lines:
        return_lines.append(preprocess(line))

    return return_lines


def preprocess(line):
    line = line.replace("\n", "")
    line = START + " " + line + " " + END
    return line


def process_test_data(sentences):
    return_list = []
    for sentence in sentences:
        return_list.append(sentence.split())
    return return_list

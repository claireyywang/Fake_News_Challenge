import collections

def count_error_types(filename):

    agree_error = collections.Counter()
    disagree_error = collections.Counter()
    discuss_error = collections.Counter()
    unrelated_error = collections.Counter()

    with open(filename, 'r') as f:
        for line in f:
             vals = line.split(" ")

             if vals[1] == 'agree':
                 agree_error.update({vals[2]: 1})
             elif vals[1] == 'disagree':
                 disagree_error.update({vals[2]: 1})
             elif vals[1] == 'discuss':
                 discuss_error.update({vals[2]: 1})
             else:
                 unrelated_error.update({vals[2]: 1})

    print("AGREE ERRORS:\n")
    print(agree_error)
    print("DISAGREE ERRORS:\n")
    print(disagree_error)
    print("DISCUSS ERRORS:\n")
    print(discuss_error)
    print("UNRELATED ERRORS:\n")
    print(unrelated_error)


if __name__ == "__main__":
    count_error_types('test_errors_extension3.txt')

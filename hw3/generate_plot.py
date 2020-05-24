from sys import argv


def plot():
    qst = handle_input(argv)


def handle_input(argv):
    """
        Handles input error
    """

    choices = ['q1', 'q2', 'q3', 'q4', 'q5']

    if len(argv) < 2:
        err_msg = 'Missing the problem number'
        exit(err_msg)
    problem = argv[1].lower()

    if problem not in choices:
        msg = 'Invalid choice: ' + 'choices = ' + ', '.join(choices)
        exit(msg)

    return problem


if __name__ == '__main__':
    plot()

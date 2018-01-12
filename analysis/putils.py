import sys
import random
import matplotlib.pyplot as plt
#import ipdb

RST = b'\x1B[0m'
RED = b'\x1B[31m'
GRN = b'\x1B[32m'
YEL = b'\x1B[33m'
BLU = b'\x1B[34m'
MAG = b'\x1B[35m'
CYN = b'\x1B[36m'
BLD = b'\x1B[1m'
REV = b'\x1B[7m'
WHT = b'\x1B[37m'



def plot_graphs(graphs, titles):
    """ Plot n graphs with the specified titles
    :param graphs: list(np.array) graphs to be plot
    :param titles: list(string) titles of the graphs
    :return:
    """
    for i, (graph, title) in enumerate(zip(graphs, titles)):

        plt.subplot(1, len(graphs), i+1)
        plt.imshow(graph)
        plt.title(title)

    plt.show()
    plt.close()


def cprint(msg, COL=YEL, end=b'\n'):
    """
    Colored print, it prints a bold black msg with a yellow background.
    :param COL: the colour of the background
    :param end: the end char, default \n
    """
    if end != b'\n':
        end = end.encode()
    sys.stdout.buffer.write(BLD+REV+COL + msg.encode() + RST + end)
    sys.stdout.buffer.flush()


def cpprint(array, idxs, length=25):
    """
    Colored print of an array, it highlights the elements in idx positions.
    :param array: the array of elements
    :param idxs: the indices of the elements to be highlighted
    :param length: the length of the rows to be displayed
    """
    if len(array) < 25:
        length = 1

    count = 1
    for i in array:

        if count-1 in idxs:
            cprint(str(i), end=' ')
        else:
            sys.stdout.buffer.write(str(i).encode()+b' ')
            sys.stdout.buffer.flush()

        if count%length == 0:
            print()
        count += 1

def condprint(array, el, length=25):
    """
    Colored print of an array, it highlights the elements in idx positions.
    :param array: the array of elements
    :param idxs: the indices of the elements to be highlighted
    :param length: the length of the rows to be displayed
    """
    if len(array) < 25:
        length = 1

    count = 1
    for i in array:

        if array[count-1] == el:
            cprint(str(i), end=' ')
        else:
            sys.stdout.buffer.write(str(i).encode()+b' ')
            sys.stdout.buffer.flush()

        if count%length == 0:
            print()
        count += 1


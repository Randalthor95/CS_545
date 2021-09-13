import numpy as np


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    l = np.array([[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3]])
    r = np.array([[4, 4],
                  [5, 5],
                  [6, 6]])
    l2 = np.array([[2, 3],
                   [2, 3],
                   [2, 3]])
    print(l @ r)
    print(l2 @ r[1:])
    print(r[:1])
    print((l2 @ r[1:]) + r[:1])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

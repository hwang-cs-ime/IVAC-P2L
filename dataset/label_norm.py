""" transform label to groundtruth(density map)"""
from scipy import integrate
import math
import numpy as np
import random


def PDF(x, u, sig):
    # f(x)
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)


# integral f(x)
def get_integrate(x_1, x_2, avg, sig):
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y


def normalize_label(y_frame, y_length):
    # y_length: total frames
    # return: normalize_label  size:nparray(y_length,)
    index_pos = []
    y_label = [0 for i in range(y_length)]
    for i in range(0, len(y_frame), 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        avg = (x_b + x_a) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1 
        if num != 1:
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[x_a + j] = y_ing
        else:
            y_label[x_a] = 1

        index_pos.append(x_a)
        index_pos.append(x_b)

        # if num != 1:
        #     for j in range(num):
        #         x_1 = x_a - 0.5 + j
        #         x_2 = x_a + 0.5 + j
        #         y_ing = get_integrate(x_1, x_2, avg, sig)
        #         y_label[x_a + j] = y_ing
        #
        #     if int(x_b - num) <= x_a:
        #         index_pos.append(x_a)
        #         index_pos.append(x_b)
        #     elif int(x_b - num) > x_a:
        #         start = random.randint(x_a, int(x_b - num))
        #         end = start + math.ceil(num)
        #         index_pos.append(start)
        #         index_pos.append(end)
        # else:
        #     assert x_a == x_b, "num!=1"
        #     y_label[x_a] = 1
        #     index_pos.append(x_a)
        #     index_pos.append(x_b)

    index_pos.extend([-1 for i in range(y_length * 5 - len(index_pos))])

    return y_label, index_pos

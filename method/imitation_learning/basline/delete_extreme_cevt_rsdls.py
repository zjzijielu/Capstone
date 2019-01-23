import numpy as np

def delete_extreme_cevt_rsdls(cevt_rsdls, cursor, T=.2):
    # this function takes in a vector of a cevt's rsdls and delete the extreme
    # values compares to cursor. When the max - min is greater than T, we get
    # rid of the extreme ones compared to cursor (the last cevt_rsdl)
    #  input:
    #         cevt_rsdls: a vector 1*x of rsdles
    #         cursor: the default rsdl center
    #         T: maximum span of the rsdl within a cevts
    #  output:
    #         new_rsdls: after deleting extreme value of cevts_rsdls, a vector 1* y,y <=x
    #         retain_ix:  cevt_rsdls(retain_ix) = new_rsdls

    # this is very ugly.. but thresholding is hard
    T = max(.2, T)
    T = min(T, .3)

    order_rsdls = np.sort(cevt_rsdls - cursor)
    ix = np.argsort(cevt_rsdls - cursor)
    tail = len(order_rsdls) - 1
    head = 0
    while order_rsdls[tail] - order_rsdls[head] > T:
        if abs(order_rsdls[tail]) > abs(order_rsdls[head]):
            tail -= 1
        else:
            head += 1
    new_rsdls = order_rsdls[head:tail+1] + cursor
    retain_ix = ix[head:tail+1]

    return new_rsdls, retain_ix

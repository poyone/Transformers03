def kai(num, sqrt, acc):
    left = 0
    right = num
    midden = num/2
    acc_now = abs(midden**sqrt - num)

    while acc_now > acc:
        if midden**sqrt > num :
            left, right = left, midden
            midden = (right + left)/2

        elif midden**sqrt < num :
            left, right = midden, right
            midden = (right + left)/2

        else:
            return midden

        acc_now = abs(midden**sqrt - num)

    return midden

a = kai(8, 4, 0.000001)
print(a)
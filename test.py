def prevPermOpt1(arr):
    num = len(arr)
    if num < 3:
        return arr.sort()

    tempmin = arr[-1]
    temp = num - 1
    mark = 0
    for i in range(num - 2, -1,-1):
        if arr[i] > tempmin:
            mark = 1
            break
        elif arr[i] < tempmin:
            tempmin = arr[i]
            temp = i
    if mark == 0:
        return arr
    n = arr[i]
    s = i

    for i in range(num - 1, s,-1):
        if arr[i] < n and arr[i - 1] < arr[i]:
            break
    arr[i], arr[s] = arr[s], arr[i]
    return arr
arr = [3,2,1]
print(prevPermOpt1(arr))


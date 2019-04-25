"""
def sorting_bs(list):
    list1 = list

    median = list[len(list) // 2]

    right = 0
    left = len(list)

    while left < right:
        if median < list[right]:


            list[right] = median
            right += 1

        else:

            left += 1

    return list

print(sorting_bs([3,8,1,5]))
"""

l = [3,1,2,9]
median = l[len(l) // 2]
print(median)
right = 0
left = len(l)
while right < left:
    if median < l[right]:
        l[right] = median
        right += 1
    else:
        left += 1

print(l)



C = "We are studying python now"
my_dict = {}
for i in C:
    if my_dict.get(i) is None:
        my_dict[i] = 1
    else:
        my_dict[i] += 1
print(my_dict)

#
# import matplotlib.pyplot as plt
#
# num_list = [0.0929933 , 0.08506372 ,0.14606655, 0.24464758, 0.06144161, 0.25290242,
#   0.06328169 ,0.05360315 ]
#
# name_list = ['Our','waiter', 'was', 'horrible', 'so', 'rude', 'and' ,'.']
# plt.bar(range(len(num_list)), num_list,tick_label=name_list)
# plt.show()


import numpy as np
import tensorflow as tf
# a = [[1,2],
#      [3,4]]
#
# c = np.sum(a,axis = 0)
# print (c)

# for i in range(0,10,2):
#      print (i)

# index = tf.range(0, 5) * 2
# with tf.Session() as sess:
#      sess.run(index)
#
# print (index)

# a = np.array([[[1,2,3]],
#      [[4,5,6]]])
# b = np.array([[[2,2,2],
#               [2,2,2]],
#               [[2,2,2],
#                [2,2,2]]])
#
# print (a.shape)
# print (b.shape)
#
# c = a * b
# print (c)

# a = np.array([[[2,2,2,7,8],
#                [2,2,2,4,5],
#                [1,2,3,4,5]],
#                [[2,2,2,3,8],
#                 [2,2,2,7,9],
#                 [3,4,5,6,7]]])
# print (a.shape)
# b = np.array([2 ,1])
# sxl = tf.reduce_max(a, 2,keep_dims=True)
# with tf.Session() as sess:
#     print (sess.run(sxl))

# a = 1 if 6 % 2 else 0
# print (a)
# class_set = set([-1,1,1,0])
# sxl = dict(zip(class_set, range(3)))
# # a = [3,4,5]
# # np.random.shuffle(a)
# print (sxl)
def change_y_to_onehot(y):
    from collections import Counter
    print (Counter(y))
    class_set = set(y)
    n_class = 3
    y_onehot_mapping = dict(zip(class_set, range(n_class)))  #{0: 0, 1: 1, -1: 2}
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

print (change_y_to_onehot([-1,-1,-1,-1,-1,1,-1,0]))
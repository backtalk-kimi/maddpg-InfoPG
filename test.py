# import os
# import numpy as np
# file_name = 'returns.pkl.npy'
# file = 'model'
# file_1 = 'simple_tag'
# current_dir = os.getcwd()
# file_name = os.path.join(current_dir, file, file_1, file_name)
# with open(file_name, 'rb') as f:
#     data = np.load(f)
# print(data)
# class ListNode:
#     def __init__(self,val):
#         self.val = val
#         self.node = None
#
# class MyLinkedList:
#
#     def __init__(self):
#         self.length = 0
#         self.next = None
#
#     def get(self, index: int) -> int:
#         if index > self.length - 1:
#             return -1
#         node = self.next
#         for i in range(index):
#             node = node.next
#         return node.val
#
#     def addAtHead(self, val: int) -> None:
#         node = ListNode(val)
#         node.next = self.next
#         self.next = node
#         self.length += 1
#
#     def addAtTail(self, val: int) -> None:
#         node = self
#         for i in range(self.length):
#             node = node.next
#         new = ListNode(val)
#         node.next = new
#         self.length += 1
#
#     def addAtIndex(self, index: int, val: int) -> None:
#         if index > self.length:
#             return
#         if index == 0:
#             self.addAtHead(val)
#         else:
#             node = self.next
#             new = ListNode(val)
#             for i in range(index - 1):
#                 node = node.next
#             new.next = node.next
#             node.next = new
#             self.length += 1
#     def deleteAtIndex(self, index: int) -> None:
#         if index > self.length - 1 or index < 0:
#             return
#         pref = self
#         for i in range(index):
#             pref = pref.next
#         node = pref.next
#         pref.next = node.next
#         self.length -= 1
#
# link = MyLinkedList()
# link.addAtHead(7)
# link.addAtHead(2)
# link.addAtHead(1)
# link.addAtIndex(3,0)
# link.deleteAtIndex(2)
# link.addAtHead(6)
# link.addAtTail(4)
# print(link.get(4))

# import numpy as np
#
# # 创建两个ndarray
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])
#
# # 使用np.concatenate()函数拼接两个ndarray
# concatenated_array = np.vstack([array1, array2])
#
# # 打印拼接后的ndarray
# print(concatenated_array.mean(axis=0))
import matplotlib.pyplot as plt

returns = [[i,i+1] for i in range(10)]
plt.figure()
plt.plot(range(len(returns)), returns)
plt.xlabel('episode * ' + str(1))
plt.ylabel('average returns')
plt.savefig('.' + '/plt.png', format='png')
plt.cla()

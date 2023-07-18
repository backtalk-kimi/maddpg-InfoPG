import re
import matplotlib.pyplot as plt
file_name = 'return.txt'
with open(file_name,'rb') as f:
    date = f.readlines()
returns = []
pattern = r'\d+\.'
for line in date:
    # print(type(line))
    line = line.decode()
    nums = re.findall(pattern, line)
    for num in nums:
        num = num.split('.')[0]
        returns.append(int(num))
plt.figure()
plt.plot(range(len(returns)), returns)
plt.xlabel('episode * ' + str(1000))
plt.ylabel('average returns')
plt.savefig('plt.png', format='png')
plt.cla()
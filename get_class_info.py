from datetime import datetime
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data_type = 1
update_type = 0
n_samples, n_classes, hidden_dim, epoch_p, lr, std = 1000, 6, 5 , 1000, 0.5, 1
root = f'D:/Applications/vscode/workspace/JMLM/outputs/'
n_classes = list(range(3, 10))
nrow = 1
ncol = 1
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
# for n_class in n_classes:
# n_class = 5
epoch = "1000"
# tmp_dir = root + f'class_{n_class}/'
tmp_dir = root + f"ICU/"
acc_list = []
test_acc_list = []
n_centers = list(range(1, 20))
for n_center in n_centers:
    tmp_center_dir = tmp_dir + f'center_{n_center}/'
    final_dir = ""
    for path in listdir(tmp_center_dir):
        if path.find('epoch' + epoch) != -1:
            final_dir = path
            break
    acc_file = tmp_center_dir + final_dir + "/acc.txt"
    f = open(acc_file, "r")
    lines = f.readlines()
    for line in lines:
        if line.find("train_acc") != -1 and line.find("vote_train_acc") == -1:
            acc = float(line.replace('train_acc = ', ''))
            acc_list.append(acc)
        if line.find("test_acc") != -1 and line.find("vote_test_acc") == -1:
            test_acc = float(line.replace('test_acc = ', ''))
            test_acc_list.append(test_acc)
    f.close()
print(f"{acc_list = }")
print(f"{test_acc_list = }")
# axi = n_class - 3
# ax.plot(n_centers, acc_list, label=f'{n_class}', marker='o')
ax.set_title("ICU")
ax.plot(n_centers, acc_list, label=f'Train', marker='o')
ax.plot(n_centers, test_acc_list, label=f'Test', marker='s', c='r')
for x, y in enumerate(acc_list):
    label = "{:.4f}".format(y)
    plt.annotate(label, # this is the text
                 (x+1,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

for x, y in enumerate(test_acc_list):
    label = "{:.4f}".format(y)
    plt.annotate(label, # this is the text
                 (x+1,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


ax.legend(loc='best')
ax.grid('on')
# ax.set_title(f'{n_class = }\n')
# ax.set_title(f'{n_class = }\n{max(acc_list)}')
# ax.set_xlabel(r"$n_centers$")
ax.set_xlabel(r"Number of $\varphi_j(\mathbf{x})$")
ax.set_ylabel(r"$Accuracy$")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))



plt.savefig(root + "acc.png")
plt.close(fig)
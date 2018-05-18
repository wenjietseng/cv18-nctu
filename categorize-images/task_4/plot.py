import matplotlib.pyplot as plt
import numpy as np
import csv

# training loss
drop_train = open('./WJNet-train-dropout.csv', 'r')
res18_train = open('./WJNet-train-res18-3232.csv', 'r')
plain_train = open('./WJNet-train0.csv', 'r')
drop_test = open('./WJNet-test-dropout.csv', 'r')
res18_test = open('./WJNet-test-res18-3232.csv', 'r')
plain_test = open('./WJNet-test0.csv', 'r')


def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta):
    loss = []
    for l in dta[:, 1]:
        loss.append(float(l))
    return loss

def get_test_err(dta):
    err = []
    for e in dta[:, 2]:
        err.append((100.0 - float(e)) / 100.0)
    return err

drop_train_loss = get_train_loss(read_table(drop_train))
res18_train_loss = get_train_loss(read_table(res18_train))
plain_train_loss = get_train_loss(read_table(plain_train))

drop_test_err = get_test_err(read_table(drop_test))
res18_test_err = get_test_err(read_table(res18_test))
plain_test_err = get_test_err(read_table(plain_test))

print(1.0 - min(plain_test_err))
print(1.0 - min(drop_test_err))
print(1.0 - min(res18_test_err))

# training loss
plt.subplots()
plt.plot(range(80), plain_train_loss, label='WJNet', color="skyblue")
plt.plot(range(80), drop_train_loss, label="WJNet (w/ dropout)", color="steelblue")
plt.plot(range(80), res18_train_loss, label='ResNet18', color='navy')

plt.legend()
plt.ylim([0., 2.8])
plt.xlabel("Epoch 1 - 80")
plt.ylabel('Training Loss')
plt.savefig("traing_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# testing error resnet 20, 56, 110
plt.subplots()
plt.plot(range(80), plain_test_err, label='WJNet', color="skyblue")
plt.plot(range(80), drop_test_err, label="WJNet (w/ dropout)", color="steelblue")
plt.plot(range(80), res18_test_err, label='ResNet18', color='navy')

plt.legend()
plt.ylim([0, 1])
plt.xlabel("Epoch 1 - 80")
plt.ylabel('Testing Error')
plt.savefig("res_testing_error.png", dpi=300, bbox_inches='tight')
plt.close()

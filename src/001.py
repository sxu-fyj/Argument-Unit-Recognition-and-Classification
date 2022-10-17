import pickle as pkl
with open('loss_log2.pkl', 'rb') as fr:
    loss_otes, loss_stances, loss_unifieds, losss = pkl.load(fr)
loss_otes, loss_stances, loss_unifieds, losss = loss_otes[::200], loss_stances, loss_unifieds[::200], losss[::200]

from matplotlib import pyplot as plt
#设置x
x=range(0,len(losss))
#设置y
y1=[]
y2=[]
y3=[]
# for i in loss_otes:
#     y1.append(i.cpu().tolist())
# for i in loss_stances:
#     y2.append(i.cpu().tolist())
for i in losss:
    y3.append(i.cpu().tolist())

# plt.plot(x,y1,color='red',label='自己',zorder=5)
# plt.plot(x,y2,color='blue',label='同事李',zorder..=10)
plt.plot(x,y3,color='green',label='同事张',zorder=15)

#y=[14,17,19,11,14,13,15,16]
#plot函数需要两个参数，一个是x一个是y
plt.show()
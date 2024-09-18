import os

os.mkdir('label_train')
path='label_train'
ncs=os.listdir('new_train')
for rank,i in enumerate(ncs):
    nm=os.listdir('new_train/'+i)
    os.mkdir(path+'/'+i)
    for j in nm:
        j=j[:-4]
        lb=open(f'{path}/{i}/{j}.txt','w',encoding='utf-8')
        lb.write(f"{rank} 0.5 0.5 1 1")
import os
f = open('train.txt','w')
filenames = os.listdir('./')
for name in filenames:
    if name.split('.')[-1]=='bmp':
        f.write(name+' '+str(0)+'\n')
f.close()
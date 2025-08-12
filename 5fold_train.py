import os
for i in range(5):
    print('Fold', i)
    cmd = 'python CLIP_CERVIX.py --model Ourmodel --bs 64 --lr 0.003 --mode 1 --fold %d' %(i+1)
    os.system(cmd)
print("Train CLIP_CERVIX ok!")
os.system('pause')

import os

def get_name_labels(dir_name):
    for parent, dir, file_names in os.walk(dir_name):
        for file_name in file_names:
            #print("parent is :", parent)
            #print ("file_name is :",file_name)
            cur_file = parent.split('\\')[-1]
            if cur_file == 'class0':
                label = 0
            elif cur_file == 'class1':
                label = 1
            elif cur_file == 'class2':
                label = 2
            elif cur_file == 'class3':
                label =3
            image_name_label = cur_file+'\\'+file_name+' '+str(label)
            yield image_name_label

def write_txt(dir_name, filename, mode='w'):
    with open(filename, mode) as f:
        for line in get_name_labels(dir_name):
            f.write(line+'\n')

if __name__ == '__main__':
    train_dir = 'dataset/train'
    train_txt_dir = 'dataset/train.txt'

    write_txt(train_dir,train_txt_dir,'w')
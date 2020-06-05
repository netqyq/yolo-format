# python3
# Pupurse: pick a subset from the whole dataset.

import glob
import array
import os

label_path = './VOC2007/labels'
anno_files = glob.glob(label_path + '/'+'*.txt')

# absolute path of the imgs
IMG_PATH = '/home/ma-user/work/trainval/VOC2007/JPEGImages/'

# total imgs will be selected
TOTAL_SELECT = 600

TEST_FILE = 'test'+str(TOTAL_SELECT)+'.txt'
TRAIN_FILE = 'train'+str(TOTAL_SELECT)+'.txt'

# total number of classes
NUM_CLASSES = 44

# 0 -> ['file1', 'file2', ...]
# 1 -> [...]
# ...
# 43 -> [...]
select_list = {}

# map(int -> int)
# save how many imgs are pick in each class
select_sign = {}

# selected_imgs is whole img set, no duplication
selected_imgs = set()

# maximum number of imgs will be picked
max_num_each = 20

# control for testing
iter_counter = 0

# initail data structure


def init_select_sign(num_classes, select_sign):
    for i in range(num_classes):
        select_sign[i] = 0


def init_select_list(num_classes, select_list):
    for i in range(num_classes):
        select_list[i] = []


def extract_class_num(line):
    fields = line.split(' ')
    return int(fields[0])


def save_file(selected_imgs, ratio=0.2):
    num_train = int(TOTAL_SELECT * (1-ratio))
    num_test = int(TOTAL_SELECT * ratio)

    print(num_train, num_test)
    imgs = list(selected_imgs)
    train = imgs[:num_train]
    test = imgs[num_train:]
    print(test)
    with open(TEST_FILE, 'w') as test_f:
        for img in test:
            line = IMG_PATH + img + '.jpg' +'\n'
            test_f.write(line)

    with open(TRAIN_FILE, 'w') as train_f:
        for img in train:
            line = IMG_PATH + img + '.jpg' + '\n'
            train_f.write(line)


if __name__ == "__main__":
    init_select_sign(NUM_CLASSES, select_sign)
    init_select_list(NUM_CLASSES, select_list)
    # print(imgs)
    for anno_file in anno_files:
        # if iter_counter > 3:
        #     break
        with open(anno_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_num = extract_class_num(line)
                if select_sign[class_num] < max_num_each:
                    basename = os.path.basename(f.name)
                    img_name = basename[:-4]
                    selected_imgs.add(img_name)
                    select_sign[class_num] += 1
                    class_list = select_list[class_num]
                    class_list.append(img_name)

        # print('read one file')
        # iter_counter += 1
        # stop check
        if len(selected_imgs) >= TOTAL_SELECT:
            break

    # print(selected_imgs)

    # print(select_list)
    # print(select_sign)

    with open('selected_imgs.txt', 'w') as imgs_f:
        for img_fname in selected_imgs:
            imgs_f.write(img_fname+'\n')    
            
    save_file(selected_imgs, ratio=0.2)

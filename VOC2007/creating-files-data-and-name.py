

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-5
Converting Traffic Signs dataset in YOLO format
File: creating-files-data-and-name.py
"""


# Creating files ts_data.data and classes.names
# for training in Darknet framework
#
# Algorithm:
# Setting up full path --> Reading file classes.txt -->
# --> Creating file classes.names -->
# --> Creating file ts_data.data
#
# Result:
# Files classes.names and ts_data.data needed to train
# in Darknet framework


"""
Start of:
Setting up full path to directory with Traffic Signs images
"""

# Full or absolute path to the folder with Traffic Signs images
# Find it with Py file getting-full-path.py
# Pay attention! If you're using Windows, yours path might looks like:
# r'\home\my_name\Downloads\ts'
# or:
# '\\home\\my_name\\Downloads\\ts'
full_path_to_images = '/Users/yq/Downloads/datasets/trainval/VOC2007/JPEGImages'

"""
End of:
Setting up full path to directory with Traffic Signs images
"""

# set data file name for train
data_file_name = 'rubbish_data.data'

"""
Start of:
Creating file classes.names
"""

# Defining counter for classes
c = 0

# Creating file classes.names from existing one classes.txt
# Pay attention! If you're using Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with open(full_path_to_images + '/' + 'classes.names', 'w') as names, \
     open(full_path_to_images + '/' + 'classes.txt', 'r') as txt:

    # Going through all lines in txt file and writing them into names file
    for line in txt:
        names.write(line)  # Copying all info from file txt to names

        # Increasing counter
        c += 1

"""
End of:
Creating file classes.names
"""


"""
Start of:
Creating file ts_data.data
"""

# Creating file ts_data.data
# Pay attention! If you're using Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with open(full_path_to_images + '/' + data_file_name , 'w') as data:
    # Writing needed 5 lines
    # Number of classes
    # By using '\n' we move to the next line
    data.write('classes = ' + str(c) + '\n')

    # Location of the train.txt file
    data.write('train = ' + full_path_to_images + '/' + 'train.txt' + '\n')

    # Location of the test.txt file
    data.write('valid = ' + full_path_to_images + '/' + 'test.txt' + '\n')

    # Location of the classes.names file
    data.write('names = ' + full_path_to_images + '/' + 'classes.names' + '\n')

    # Location where to save weights
    data.write('backup = backup')

"""
End of:
Creating file ts_data.data
"""

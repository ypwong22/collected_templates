###############################################################################
# List only directories under the given directory.
# os.walk() walks through each layer of children-directories from top down, 
#           or set topdown=False to go from bottom up.
###############################################################################
import os
path_root = '/my/path'

# yields '/my/path'
next(os.walk(path_var))[0]

# yields the directories in '/my/path'
next(os.walk(path_var))[1]

# yields the files in '/my/path'
next(os.walk(path_var))[2]

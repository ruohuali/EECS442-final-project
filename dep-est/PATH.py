import os

# DATA_PATH = "../../data/data_depth_self_compiled"
# TRAIN_PATH = os.path.join(DATA_PATH, "training")
# TRAIN_RGB_PATH = os.path.join(TRAIN_PATH, "rgb")
# TRAIN_LABEL_PATH = os.path.join(TRAIN_PATH, "label")
# TRAIN_RGB_PATHS = [os.path.join(TRAIN_RGB_PATH, "0001"),
#                 os.path.join(TRAIN_RGB_PATH, "0002"),
#                 os.path.join(TRAIN_RGB_PATH, "0009"),
#                 os.path.join(TRAIN_RGB_PATH, "0011"),
#                 os.path.join(TRAIN_RGB_PATH, "0017"),
#                 os.path.join(TRAIN_RGB_PATH, "0018"),
#                 os.path.join(TRAIN_RGB_PATH, "0048"),
#                 os.path.join(TRAIN_RGB_PATH, "0051")]

# TRAIN_DEP_PATHS = [os.path.join(TRAIN_LABEL_PATH, "0001"),
#                 os.path.join(TRAIN_LABEL_PATH, "0002"),
#                 os.path.join(TRAIN_LABEL_PATH, "0009"),
#                 os.path.join(TRAIN_LABEL_PATH, "0011"),
#                 os.path.join(TRAIN_LABEL_PATH, "0017"),
#                 os.path.join(TRAIN_LABEL_PATH, "0018"),
#                 os.path.join(TRAIN_LABEL_PATH, "0048"),
#                 os.path.join(TRAIN_LABEL_PATH, "0051")]

# TEST_PATH = os.path.join(DATA_PATH, "testing")
# TEST_RGB_PATH = os.path.join(TEST_PATH, "rgb")
# TEST_DEP_PATH = os.path.join(TEST_PATH, "label")


# DATA_PATH = "/home/ruohuali/Desktop/depth-estimation/data_depth_selection/depth_selection/"
# TRAIN_PATH = os.path.join(DATA_PATH, "test_depth_completion_anonymous")
# TRAIN_RGB_PATH = os.path.join(TRAIN_PATH, "image")
# TRAIN_DEP_PATH = os.path.join(TRAIN_PATH, "velodyne_raw")
# TRAIN_RGB_PATHS = [TRAIN_RGB_PATH]
# TRAIN_DEP_PATHS = [TRAIN_DEP_PATH]    


DATA_PATH = "/home/ruohuali/Desktop/depth-estimation/DIODE_selection"
TRAIN_PATH = os.path.join(DATA_PATH, "scan_00199") 
TRAIN_PATHS = [TRAIN_PATH]
TEST_PATHS = [TRAIN_PATH]
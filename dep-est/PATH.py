import os

DATA_PATH = "/home/ruohuali/Desktop/depth-estimation/DIODE_selection"
# TRAIN_PATH = os.path.join(DATA_PATH, "scan_00199") 
TRAIN_PATHS = [os.path.join(DATA_PATH, "scan_00193"),
               os.path.join(DATA_PATH, "scan_00194"),
               os.path.join(DATA_PATH, "scan_00195"),
               os.path.join(DATA_PATH, "scan_00196"),
               os.path.join(DATA_PATH, "scan_00197"),
               os.path.join(DATA_PATH, "scan_00198"),
               os.path.join(DATA_PATH, "scan_00200"),
               os.path.join(DATA_PATH, "scan_00201")]
TEST_PATHS = [os.path.join(DATA_PATH, "scan_00199"),
              os.path.join(DATA_PATH, "scan_00202")]


KITTI_DEP_DATA_PATH = "/home/u00uaj36zf9W01H191357/test/depth_selection"
KITTI_TRAIN_RGB_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_selection_cropped/image"),
                         os.path.join(KITTI_DEP_DATA_PATH, "test_depth_completion_anonymous/image")]
KITTI_TRAIN_LABEL_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_selection_cropped/velodyne_raw"),
                           os.path.join(KITTI_DEP_DATA_PATH, "test_depth_completion_anonymous/velodyne_raw"),]

KITTI_TEST_RGB_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_selection_cropped/test_image")]
KITTI_TEST_LABEL_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_seslection_cropped/test_velodyne_raw")]


KITTI_SEM_DATA_PATH = "/home/u00uaj36zf9W01H191357/test/data_semantics"
KITTI_SEM_TRAIN_RGB_PATHS = [os.path.join(KITTI_SEM_DATA_PATH, "training/image_2")]
KITTI_SEM_TRAIN_LABEL_PATHS = [os.path.join(KITTI_SEM_DATA_PATH, "training/semantic")]

KITTI_SEM_TEST_RGB_PATHS = [os.path.join(KITTI_SEM_DATA_PATH, "testing/image_2")]
KITTI_SEM_TEST_LABEL_PATHS = KITTI_SEM_TRAIN_LABEL_PATHS
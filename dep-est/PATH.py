# import os
#
# DATA_PATH = "/home/ruohuali/Desktop/depth-estimation/DIODE_selection"
# # TRAIN_PATH = os.path.join(DATA_PATH, "scan_00199")
# TRAIN_PATHS = [os.path.join(DATA_PATH, "scan_00193"),
#                os.path.join(DATA_PATH, "scan_00194"),
#                os.path.join(DATA_PATH, "scan_00195"),
#                os.path.join(DATA_PATH, "scan_00196"),
#                os.path.join(DATA_PATH, "scan_00197"),
#                os.path.join(DATA_PATH, "scan_00198"),
#                os.path.join(DATA_PATH, "scan_00200"),
#                os.path.join(DATA_PATH, "scan_00201")]
# TEST_PATHS = [os.path.join(DATA_PATH, "scan_00199"),
#               os.path.join(DATA_PATH, "scan_00202")]
#
#
# KITTI_DEP_DATA_PATH = "../../data/data_depth_selection/depth_selection"
# KITTI_TRAIN_RGB_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_selection_cropped/image"),
#                          os.path.join(KITTI_DEP_DATA_PATH, "test_depth_completion_anonymous/image"),
#                          "../../data/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync\image_02\data",
#                          "../../data/2011_09_26_drive_0011_sync/2011_09_26/2011_09_26_drive_0011_sync\image_02\data",
#                          "../../data/2011_09_26_drive_0017_sync/2011_09_26/2011_09_26_drive_0017_sync\image_02\data",
#                          "../../data/2011_09_26_drive_0018_sync/2011_09_26/2011_09_26_drive_0018_sync\image_02\data",]
# KITTI_TRAIN_LABEL_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_selection_cropped/velodyne_raw"),
#                            os.path.join(KITTI_DEP_DATA_PATH, "test_depth_completion_anonymous/velodyne_raw"),
#                            "../../data/data_depth_annotated/train/2011_09_26_drive_0001_sync\proj_depth\groundtruth\image_02",
#                            "../../data/data_depth_annotated/train/2011_09_26_drive_0011_sync\proj_depth\groundtruth\image_02",
#                            "../../data/data_depth_annotated/train/2011_09_26_drive_0017_sync\proj_depth\groundtruth\image_02",
#                            "../../data/data_depth_annotated/train/2011_09_26_drive_0018_sync\proj_depth\groundtruth\image_02",]
#
# KITTI_TEST_RGB_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_selection_cropped/test_image")]
# KITTI_TEST_LABEL_PATHS = [os.path.join(KITTI_DEP_DATA_PATH, "val_seslection_cropped/test_velodyne_raw")]
#
#
# KITTI_SEM_DATA_PATH = "../../data/data_semantics"
# KITTI_SEM_TRAIN_RGB_PATHS = [os.path.join(KITTI_SEM_DATA_PATH, "training/image_2")]
# KITTI_SEM_TRAIN_LABEL_PATHS = [os.path.join(KITTI_SEM_DATA_PATH, "training/semantic")]
#
# KITTI_SEM_TEST_RGB_PATHS = [os.path.join(KITTI_SEM_DATA_PATH, "testing/image_2")]
# KITTI_SEM_TEST_LABEL_PATHS = KITTI_SEM_TRAIN_LABEL_PATHS
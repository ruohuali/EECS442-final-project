import os

DIODE_DATA_PATH__ = "/home/ruohuali/Desktop/depth-estimation/DIODE_selection"
# TRAIN_PATH = os.path.join(DATA_PATH, "scan_00199") 
DIODE_TRAIN_PATHS__ = ["scan_00193",
                       "scan_00194",
                       "scan_00195",
                       "scan_00196",
                       "scan_00197",
                       "scan_00198",
                       "scan_00200",
                       "scan_00201",
                       "scan_00199",
                       "scan_00202"]

# KITTI_DEP_DATA_PATH__ = "../../data"
KITTI_DEP_DATA_PATH__ = "../../data"
KITTI_DEP_TRAIN_RGB_PATHS__ = ["data_depth_selection/depth_selection/val_selection_cropped/image",
                               "data_depth_selection/depth_selection/test_depth_completion_anonymous/image",]
                            #    "2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync\image_02\data",
                            #    "2011_09_26_drive_0011_sync/2011_09_26/2011_09_26_drive_0011_sync\image_02\data",
                            #    "2011_09_26_drive_0017_sync/2011_09_26/2011_09_26_drive_0017_sync\image_02\data",
                            #    "2011_09_26_drive_0018_sync/2011_09_26/2011_09_26_drive_0018_sync\image_02\data", ]
KITTI_DEP_TRAIN_LABEL_PATHS__ = ["data_depth_selection/depth_selection/val_selection_cropped/velodyne_raw",
                                 "data_depth_selection/depth_selection/test_depth_completion_anonymous/velodyne_raw",]
                                #  "data_depth_annotated/train/2011_09_26_drive_0001_sync\proj_depth\groundtruth\image_02",
                                #  "data_depth_annotated/train/2011_09_26_drive_0011_sync\proj_depth\groundtruth\image_02",
                                #  "data_depth_annotated/train/2011_09_26_drive_0017_sync\proj_depth\groundtruth\image_02",
                                #  "data_depth_annotated/train/2011_09_26_drive_0018_sync\proj_depth\groundtruth\image_02", ]

KITTI_SEM_DATA_PATH__ = "../data/data_semantics"
print(os.getcwd())
KITTI_SEM_TRAIN_RGB_PATHS__ = ["training/image_2", "testing/image_2"]
KITTI_SEM_TRAIN_LABEL_PATHS__ = ["training/semantic", "testing/image_2"]

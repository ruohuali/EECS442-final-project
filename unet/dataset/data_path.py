import os

from .PATH__ import DIODE_DATA_PATH__, DIODE_TRAIN_PATHS__, \
    KITTI_DEP_DATA_PATH__, KITTI_DEP_TRAIN_RGB_PATHS__, KITTI_DEP_TRAIN_LABEL_PATHS__, \
    KITTI_SEM_DATA_PATH__, KITTI_SEM_TRAIN_RGB_PATHS__, KITTI_SEM_TRAIN_LABEL_PATHS__


def initDatasetPath(diode_data_path, diode_train_paths,
                    kitti_dep_data_path, kitti_dep_train_rgb_paths, kitti_dep_train_label_paths,
                    kitti_sem_data_path, kitti_sem_train_rgb_paths, kitti_sem_train_label_paths):
    kitti_dep_train_rgb_paths = [os.path.join(kitti_dep_data_path, path) for path in kitti_dep_train_rgb_paths]
    kitti_dep_train_label_paths = [os.path.join(kitti_dep_data_path, path) for path in kitti_dep_train_label_paths]

    kitti_sem_train_rgb_paths = [os.path.join(kitti_sem_data_path, path) for path in kitti_sem_train_rgb_paths]
    kitti_sem_train_label_paths = [os.path.join(kitti_sem_data_path, path) for path in kitti_sem_train_label_paths]

    diode_train_paths = [os.path.join(diode_data_path, path) for path in diode_train_paths]

    return [kitti_dep_train_rgb_paths, kitti_dep_train_label_paths,
            kitti_sem_train_rgb_paths, kitti_sem_train_label_paths, diode_train_paths]


PATHS = initDatasetPath(DIODE_DATA_PATH__,
                        DIODE_TRAIN_PATHS__,
                        KITTI_DEP_DATA_PATH__,
                        KITTI_DEP_TRAIN_RGB_PATHS__,
                        KITTI_DEP_TRAIN_LABEL_PATHS__,
                        KITTI_SEM_DATA_PATH__,
                        KITTI_SEM_TRAIN_RGB_PATHS__,
                        KITTI_SEM_TRAIN_LABEL_PATHS__)

KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, DIODE_TRAIN_PATHS = PATHS

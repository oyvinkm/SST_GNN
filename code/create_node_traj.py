from dataprocessing.utils.loading import constructDatasetFolders, find_trajectory_nodes
from loguru import logger
if __name__ == "__main__":
    logger.success('Finding trajectories')
    find_trajectory_nodes()
    logger.success('Constructing Dataset Folders train/test/val/')
    constructDatasetFolders(same_nodes = 'same_nodes.json', choose='max')
    logger.success('Done constructing <3')

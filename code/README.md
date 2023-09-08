
# NB: ADD code/cylinder_flow/ to .gitignore
## Folder strucuture
To obtain the `cylinder_flow` dataset, please adhere to the following structure of the code directory. 
The data will not be added to git as it's way too big. 
You can download the preprocessed data from: [here](https://drive.google.com/drive/folders/1QANENxeWRVBs2TZ8SQ5CGuHo27i95WtO)

    ├── code
    │   ├── data
    │   │    ├── cylinder_flow
    │   │    │   ├── test.h5
    │   │    │   ├── train.h5
    │   │    │   ├── valid.h5
    │   │    ├── preprocessed
    │   │    │   ├── meshgraphnets_miniset5traj_vis.pt
    │   │    │   ├── meshgraphnets_miniset30traj5ts_vis.pt
    │   │    │   ├── meshgraphnets_miniset100traj25ts_vis.pt
    │   │    │   ├── test_processed_set.pt
    │   │    │   ├── valid.h5
    │   ├── datasets
    └── ├──autoencoder

*Cylinder flow data* 
Data is represented as: 

    `[Data(x=[1923, 11], edge_index=[2, 11070], edge_attr=[11070, 3], y=[1923, 2], p=[1923, 1], cells=[3612, 3], mesh_pos=[1923, 2])]`

import geopandas as gpd
import pandas as pd
import uuid

def assign_training_classes(gdf_trees: gpd.GeoDataFrame,
                            df_groups: pd.DataFrame,
                            df_classes: pd.DataFrame,
                            genus_col: str = 'genus',
                            coniferous_col: str = 'Coniferous') -> gpd.GeoDataFrame:
    """
    Assigns a `training_class` column to the GeoDataFrame `gdf_trees` based on:
    - If genus is coniferous → class 9
    - If genus in training classes → assigned class ID from df_classes
    - Else → class 8 (Other Deciduous)
    """

    # Merge coniferous info into tree GeoDataFrame
    gdf_trees = gdf_trees.merge(
        df_groups[['Genus', coniferous_col]],
        left_on=genus_col,
        right_on='Genus',
        how='left'
    ).drop(columns=['Genus'])

    # Create genus-to-class mapping
    class_map = dict(zip(df_classes['genus'], df_classes['fid']))

    # Function to assign training class per row
    def _assign(row):
        if row[coniferous_col] == 'Yes':
            return 9
        elif row[genus_col] in class_map:
            return class_map[row[genus_col]]
        else:
            return 8

    # Apply assignment
    gdf_trees['training_class'] = gdf_trees.apply(_assign, axis=1)
    gdf_trees['uuid'] = [uuid.uuid4() for f in gdf_trees.index]
    return gdf_trees[['uuid','geometry','genus',  'training_class']]

def generate_training_labels(trees_path:str = 'data/tree_data_greehill.gpkg',
                            groups_path: str = 'data/genus_summary_by_city_groups.csv',
                            labels_path: str = 'conf/model/tree_labels.csv') -> gpd.GeoDataFrame:
    """
    Generate training labels for tree species in the GeoDataFrame.
    Assigns a `training_class` column based on genus and coniferous status.
    """
    gdf_trees = gpd.read_file(trees_path)
    df_groups = pd.read_csv(groups_path)
    df_classes = pd.read_csv(labels_path)

    labeled_gdf = assign_training_classes(gdf_trees, df_groups, df_classes)
    labeled_gdf.to_file('data/tree_labels.gpkg', driver='GPKG')
    return labeled_gdf

if __name__ == "__main__":
    gdf_labeled = generate_training_labels()
    print(gdf_labeled.head())
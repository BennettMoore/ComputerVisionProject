import numpy as np
import open3d as o3d
import os

def background_subtract(curr, prev):
    min_distance = 0.001
    if prev is not None:
        distances = curr.compute_point_cloud_distance(prev)
        distances = np.asarray(distances)
        ind = np.where(distances > min_distance)[0]
        return curr.select_by_index(ind)
    return curr


def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ground_threshold = 0.2
    height_threshold = 3
    min_points = 2
    clustering_epsilon = 2.2
    directory_path = os.path.join(os.getcwd(), 'dataset', 'PointClouds')
    prev = None
    for i in range(500):
        file_path = os.path.join(directory_path, f'{i}.pcd')
        pointcloud = o3d.io.read_point_cloud(file_path)
        # First handle background subtraction
        result = background_subtract(pointcloud, prev)
        # Then, use height thresholding to get rid of any points not belonging to cars
        points = np.asarray(result.points)
        filtered_points = points[points[:, 2] <= height_threshold]
        filtered_points = filtered_points[filtered_points[:, 2] >= ground_threshold]
        prev = pointcloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        labels = np.array(filtered_pcd.cluster_dbscan(clustering_epsilon, min_points))
        max_label = labels.max()
        print(f"{i} has {max_label + 1} clusters")
        if prev is not None: vis.remove_geometry(prev)
        vis.poll_events()
        vis.update_renderer()
        vis.add_geometry(filtered_pcd)
    
if __name__ == '__main__':
    main()
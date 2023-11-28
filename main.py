import numpy as np
import open3d as o3d
import os

def main():
    height_threshold = 3
    directory_path = os.path.join(os.getcwd(), 'dataset', 'PointClouds')
    for filename in os.listdir(os.path.join(os.getcwd(), r'dataset\PointClouds')):
        file_path = os.path.join(directory_path, filename)
        pointcloud = o3d.io.read_point_cloud(file_path)
        _, inliers = pointcloud.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
        pointcloud = pointcloud.select_by_index(inliers, invert=True)
        points = np.asarray(pointcloud.points)
        filtered_points = points[points[:, 2] <= height_threshold]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        o3d.visualization.draw_geometries([filtered_pcd])
if __name__ == '__main__':
    main()
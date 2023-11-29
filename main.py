import csv
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
    starting_id = 146
    file_header = ['vehicle_id','position_x','position_y','position_z','mvec_x','mvec_y','mvec_z','bbox_x_min','bbox_x_max','bbox_y_min','bbox_y_max','bbox_z_min','bbox_z_max']
    ground_threshold = 0.2
    height_threshold = 3
    min_points = 2
    clustering_epsilon = 2.2
    directory_path = os.path.join(os.getcwd(), 'dataset', 'PointClouds')
    prev = None
    prev_cluster = None
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
        # Debug statement to view number of clusters
        print(f"{i} has {max_label + 1} clusters")
        if prev is not None: vis.remove_geometry(prev)
        vis.poll_events()
        vis.update_renderer()
        vis.add_geometry(filtered_pcd)
        clusters = []
        midpoints = []
        csv_row = []
        # Include this for now, hopefully we can figure out tracking on other frames
        if i == 3:
            with open(f'frame_{i}.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(file_header)
                for j in range(6):
                    this_cluster = []
                    for k in range(len(labels)):
                        if j == labels[k]:
                            this_cluster.append([filtered_points[k][0], filtered_points[k][1], filtered_points[k][2]])
                    num_points = len(this_cluster)
                    x_sum = sum(point[0] for point in this_cluster)
                    y_sum = sum(point[1] for point in this_cluster)
                    z_sum = sum(point[2] for point in this_cluster)

                    midpoint = (x_sum / num_points, y_sum / num_points, z_sum / num_points)
                    clusters.append(this_cluster)
                    midpoints.append(midpoint)
                midpoints = sorted(midpoints, key=lambda point: point[0])
                curr_id = starting_id
                #csv_writer.writerow()
            #o3d.visualization.draw_geometries([filtered_pcd])
    
if __name__ == '__main__':
    main()
import csv
import numpy as np
import open3d as o3d
import os


# A set of distinct colors to distinguish different clusters
CLUSTER_COLORS = np.array([
    [0.75,0.75,0.75], # Gray
    [1.,0.,0.], # Red
    [0.,1.,0.], # Green
    [0.,0.,1.], # Blue
    [1.,1.,0.], # Yellow
    [1.,0.,1.], # Magenta
    [0.,1.,1.], # Cyan 
    [0.5,1.,0.], # Lime
    [0.5,0.,1.], # Purple
    [1.,0.5,0.], # Orange
    [0.,0.5,1.], # Sky Blue
    [0.,1.,0.75], # Turquoise
    [1.,0.5,0.75], # Pink
    [0.5,0.5,1.], # Lavender
])

def background_subtract(curr, prev):
    min_distance = 0.001
    if prev is not None:
        distances = curr.compute_point_cloud_distance(prev)
        distances = np.asarray(distances)
        ind = np.where(distances > min_distance)[0]
        return curr.select_by_index(ind)
    return curr

def color_clusters(labels):
    colors = np.zeros((labels.size, 3), dtype=np.float64)
    for i in range(labels.size):
        label = labels[i]+1 # Add 1 to account for '-1' labels
        index = label % CLUSTER_COLORS.shape[0]
        colors[i] = CLUSTER_COLORS[index-1]
        darkness = label // CLUSTER_COLORS.shape[0] # If there are more clusters than colors, slightly darken duplicate colors to distinguish them
        colors[i] = colors[i] - (colors[i] / max(1, CLUSTER_COLORS.shape[0] - darkness + 1))
    return colors
    
    

def main():
    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    num_cars = 6
    starting_index = 0
    ending_index = 499
    starting_id = 141 # He used this in the CSV
    file_header = ['vehicle_id','position_x','position_y','position_z','mvec_x','mvec_y','mvec_z','bbox_x_min','bbox_x_max','bbox_y_min','bbox_y_max','bbox_z_min','bbox_z_max']
    ground_threshold = 0.2
    height_threshold = 3
    min_points = 2
    clustering_epsilon = 2
    directory_path = os.path.join(os.getcwd(), 'dataset', 'PointClouds')
    prev = None
    stable_flag = False
    prev_cluster = None
    for i in range(starting_index, ending_index + 1):
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
        # print(labels)
        filtered_pcd.colors = o3d.utility.Vector3dVector(color_clusters(labels))
        # Debug statement to view number of clusters
        print(f"Frame {i} has {max_label + 1} clusters")

        #if prev is not None: vis.remove_geometry(prev)
        #vis.poll_events()
        #vis.update_renderer()
        #vis.add_geometry(filtered_pcd)
        clusters = []
        midpoints = []
        csv_row = []
        # Include this for now, so that we start processing when we have the first stable frame (if he grades it with another set of pointclouds)
        if max_label + 1 == num_cars and stable_flag is False:
            stable_flag = True
            with open(f'frame_{i}.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(file_header)
                for j in range(num_cars):
                    this_cluster = []
                    for k in range(len(labels)):
                        if j == labels[k]+1:
                            this_cluster.append([filtered_points[k][0], filtered_points[k][1], filtered_points[k][2]])
                    num_points = len(this_cluster)
                    x_sum = sum(point[0] for point in this_cluster)
                    y_sum = sum(point[1] for point in this_cluster)
                    z_sum = sum(point[2] for point in this_cluster)

                    midpoint = (x_sum / num_points, y_sum / num_points, z_sum / num_points)
                    clusters.append(this_cluster)
                    midpoints.append(midpoint)
                midpoints = sorted(midpoints, key=lambda point: point[0])
                # print(midpoints)
                curr_id = starting_id
                # csv_writer.writerow()
        o3d.visualization.draw_geometries([filtered_pcd])
    
if __name__ == '__main__':
    main()
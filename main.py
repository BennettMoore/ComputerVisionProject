import csv
import numpy as np
import open3d as o3d
import os

# TODO Set me to False before turning in
SHOW_EVERYTHING = True # Debugging feature to help give each cluster more context in the total point cloud

# A set of distinct colors to distinguish different clusters
CLUSTER_COLORS = np.array([
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
    [0.75,0.75,0.75], # Gray
])
NUM_CARS = 6
STARTING_INDEX = 0
ENDING_INDEX = 499
STARTING_ID = 141 # He used this in the CSV
FILE_HEADER = ['vehicle_id','position_x','position_y','position_z','mvec_x','mvec_y','mvec_z','bbox_x_min','bbox_x_max','bbox_y_min','bbox_y_max','bbox_z_min','bbox_z_max']
HEIGHT_THRESHOLD = 3
MIN_POINTS = 2
CLUSTERING_EPSILON = 2
DIR_PATH = os.path.join(os.getcwd(), 'dataset', 'PointClouds')

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

    prev = None
    stable_flag = False
    prev_midpoints = None
    for i in range(STARTING_INDEX, ENDING_INDEX + 1):
        file_path = os.path.join(DIR_PATH, f'{i}.pcd')
        pointcloud = o3d.io.read_point_cloud(file_path)
        # First handle background subtraction
        result = background_subtract(pointcloud, prev)
        # Then, use height thresholding to get rid of any points not belonging to cars
        points = np.asarray(result.points)
        filtered_points = points[points[:, 2] <= HEIGHT_THRESHOLD]
        
        prev = pointcloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        labels = np.array(filtered_pcd.cluster_dbscan(CLUSTERING_EPSILON, MIN_POINTS))
        max_label = labels.max()
        # print(labels)
        filtered_pcd.colors = o3d.utility.Vector3dVector(color_clusters(labels))
        if SHOW_EVERYTHING:
            total_points = np.concatenate((np.asarray(filtered_pcd.points), np.asarray(pointcloud.points)), axis=0)
            total_colors = np.concatenate((np.asarray(filtered_pcd.colors), np.zeros((np.asarray(pointcloud.points).shape[0], 3), dtype=np.float64)), axis=0)
            total_pcd = o3d.geometry.PointCloud()
            total_pcd.points = o3d.utility.Vector3dVector(total_points)
            total_pcd.colors = o3d.utility.Vector3dVector(total_colors)
        
        
        # Debug statement to view number of clusters
        print(f"Frame {i} has {max_label + 1} clusters")

        #if prev is not None: vis.remove_geometry(prev)
        #vis.poll_events()
        #vis.update_renderer()
        #vis.add_geometry(filtered_pcd)
        midpoints = {}
        mvecs = {}
        bboxes = {}
        # Include this for now, so that we start processing when we have the first stable frame (if he grades it with another set of pointclouds)
        if max_label + 1 == NUM_CARS and stable_flag is False:
            stable_flag = True
            with open(f'perception_results/frame_{i}.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(FILE_HEADER)
                for j in range(NUM_CARS-1, -1, -1):
                    this_cluster = []
                    for k in range(len(labels)):
                        if j == labels[k]:
                            this_cluster.append([filtered_points[k][0], filtered_points[k][1], filtered_points[k][2]])
                    num_points = len(this_cluster)
                    curr_id = STARTING_ID + j
                    
                    x_sum = sum(point[0] for point in this_cluster)
                    y_sum = sum(point[1] for point in this_cluster)
                    z_sum = sum(point[2] for point in this_cluster)
                    x_min = min(point[0] for point in this_cluster)
                    y_min = min(point[1] for point in this_cluster)
                    z_min = min(point[2] for point in this_cluster)
                    x_max = max(point[0] for point in this_cluster)
                    y_max = max(point[1] for point in this_cluster)
                    z_max = max(point[2] for point in this_cluster)
                    

                    midpoint = np.asarray((x_sum / num_points, y_sum / num_points, z_sum / num_points))
                    bbox = np.asarray((midpoint[0] - x_min, x_max - midpoint[0],midpoint[1] - y_min, y_max - midpoint[1], midpoint[2] - z_min, z_max - midpoint[2]))
                    #print(bbox_z_max)
                    #print(bbox_z_min)
                    
                    csv_writer.writerow([curr_id, midpoint[0], midpoint[1], midpoint[2], 0., 0., 0., bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
                    midpoints[curr_id] = midpoint
                    mvecs[curr_id] = np.asarray((0., 0., 0.))
                    bboxes[curr_id] = bbox
                prev_midpoints = midpoints
                prev_mvecs = mvecs
                prev_bboxes = bboxes
            for j in range(STARTING_INDEX, i):
                with open(f'perception_results/frame_{i}.csv', 'r', newline='') as csv_file:
                    with open(f'perception_results/frame_{j}.csv', 'w', newline='') as old_csv_file:
                        reader = csv.reader(csv_file)
                        prev_data = list(reader)
                        writer = csv.writer(old_csv_file)
                        writer.writerows(prev_data)

        if stable_flag:
            with open(f'perception_results/frame_{i}.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(FILE_HEADER)
                for j in range(NUM_CARS-1, -1, -1):
                    this_cluster = []
                    for k in range(len(labels)):
                        if j == labels[k]:
                            this_cluster.append([filtered_points[k][0], filtered_points[k][1], filtered_points[k][2]])
                    num_points = len(this_cluster)
                    curr_id = STARTING_ID + j
                    if num_points == 0: # Couldn't find cluster this time, so guess based on past info
                        midpoint = prev_midpoints[curr_id] + prev_mvecs[curr_id]
                        mvec = prev_mvecs[curr_id]
                        bbox = prev_bboxes[curr_id]
                    else:    
                        x_sum = sum(point[0] for point in this_cluster)
                        y_sum = sum(point[1] for point in this_cluster)
                        z_sum = sum(point[2] for point in this_cluster)
                        x_min = min(point[0] for point in this_cluster)
                        y_min = min(point[1] for point in this_cluster)
                        z_min = min(point[2] for point in this_cluster)
                        x_max = max(point[0] for point in this_cluster)
                        y_max = max(point[1] for point in this_cluster)
                        z_max = max(point[2] for point in this_cluster)
                        

                        midpoint = np.asarray((x_sum / num_points, y_sum / num_points, z_sum / num_points))
                        bbox = np.asarray((midpoint[0] - x_min, x_max - midpoint[0],midpoint[1] - y_min, y_max - midpoint[1], midpoint[2] - z_min, z_max - midpoint[2]))
                        
                        mvec = midpoint - prev_midpoints[curr_id]
                        
                        #print(bbox_z_max)
                        #print(bbox_z_min)
                        
                    csv_writer.writerow([curr_id, midpoint[0], midpoint[1], midpoint[2], mvec[0], mvec[1], mvec[1], bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
                    midpoints[curr_id] = midpoint
                    mvecs[curr_id] = mvec
                    bboxes[curr_id] = bbox
                    # This is where we would try thge algo I talked about 
                    
                prev_midpoints = midpoints
                prev_mvecs = mvecs
                prev_bboxes = bboxes
                    
            
        # if SHOW_EVERYTHING: o3d.visualization.draw_geometries([total_pcd])
        # else: o3d.visualization.draw_geometries([filtered_pcd])
    
if __name__ == '__main__':
    main()
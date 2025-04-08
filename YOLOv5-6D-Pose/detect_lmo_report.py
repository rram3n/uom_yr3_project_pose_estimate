import argparse
from re import I
import time
from pathlib import Path
import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, \
    scale_coords, set_logging, increment_path, retrieve_image
from utils.torch_utils import select_device, time_synchronized
from utils.pose_utils import box_filter, get_3D_corners, pnp, get_camera_intrinsic, MeshPly
import json
from scipy.spatial.transform import Rotation  # Add this import at the top
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import pdist

def detect(save_img=False):
    # source = './data/lmo_report_test/ten_img/rgb'
    # gt_path = "./data/lmo_report_test/ten_img/gt"

    source = './data/lmo_report_test/full_img_set/rgb'
    gt_path = "./data/lmo_report_test/full_img_set/gt"

    view_img = 0
    imgsz = 640
    cam_intrinsics = 'configs/linemod/linemod_camera.json'
    #opt.source, opt.weights, opt.view_img, opt.img_size, opt.mesh_data, opt.static_camera

    weights = './weights/LINEMOD/can/weights/best.pt'
    mesh_data = './data/LINEMOD_updated/can/can.ply'
    desired_object = 'obj_000005'

    # obj_dict = {
    #     "obj_000001": "ape",
    #     "obj_000005": "wateringcan", (called can here)
    #     "obj_000006": "cat",
    #     "obj_000008": "drill", (caller driller here)
    #     "obj_000009": "duck",
    #     "obj_000010": "eggbox",
    #     "obj_000011": "glue",
    #     "obj_000012": "holepuncher"
    # }


    # Define output path
    OUTPUT_BASE_PATH = './output/lmo_report_test'
    run_id = 0
    OUTPUT_PATH = f"{OUTPUT_BASE_PATH}_{run_id}"
    while os.path.exists(OUTPUT_PATH):
        if not os.listdir(OUTPUT_PATH):  # If folder is empty, reuse it
            abs_output_path = os.path.abspath(OUTPUT_PATH)
            print(f"Reusing existing empty folder: {OUTPUT_PATH}")
            print("output path = ", abs_output_path)
            break
        run_id += 1
        OUTPUT_PATH = f"{OUTPUT_BASE_PATH}_{run_id}"

    # Create the folder only if it does not exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        abs_output_path = os.path.abspath(OUTPUT_PATH)
        print("output path = ", abs_output_path)

    # Replace save_dir initialization with OUTPUT_PATH
    save_dir = Path(OUTPUT_PATH)

    # Initialize
    set_logging()

    #device = select_device(opt.device) # uses CPU

    device = select_device('0') # uses GPU

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    torch.save(model.state_dict(), "state_dict_model.pt")
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if cam_intrinsics:
        with open(cam_intrinsics) as f:
            cam_intrinsics = yaml.load(f, Loader=yaml.FullLoader)

        dtx = np.array(cam_intrinsics["distortion"])
        mtx = np.array(cam_intrinsics["intrinsic"])

        fx = mtx[0,0]
        fy = mtx[1,1]
        u0 = mtx[0,2]
        v0 = mtx[1,2]


        internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)   

    save_img = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    
    mesh       = MeshPly(mesh_data)
    vertices   = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D  = get_3D_corners(vertices)

    # edges_corners = [[0, 1], [0, 3], [0, 7], [1, 2], [1, 6], [2, 3], [2, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7]]
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    colormap      = np.array(['r', 'g', 'b', 'c', 'm', 'y',  'k', 'w','xkcd:sky blue' ])

    # run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # add timing file initialization
    inference_times = []
    timing_file = os.path.join(save_dir, 'timing.txt')
    with open(timing_file, 'w') as f:
        # Write table header and separator with consistent column widths
        f.write("| {:^6} | {:^12} | {:^7} |\n".format("Frame", "Time (ms)", "FPS"))
        f.write("|{:=^8}|{:=^14}|{:=^9}|\n".format("", "", ""))

    predictions = []
    t0 = time.time()
    count = 0

    # Create images subdirectory
    images_dir = os.path.join(OUTPUT_PATH, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Initialize list to store estimated poses
    estimated_poses = []
    gt_poses = []
    failed_images = []

    # Read GT Pose info from json file
    gt_pose_info = read_obj8_poses(gt_path, desired_object)

    # inference loop
    for path, img, im0s, intrinsics, shapes in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Compute intrinsics
        if cam_intrinsics is None:
            fx, fy, det_height, u0, v0, im_native_width, im_native_height = intrinsics
            # fx, fy  = # calculate_focal_length(float(focal_len), int(im_native_width), int(im_native_height), float(det_width), float(det_height))
            internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Inference
        t1 = time_synchronized()
        pred, train_out = model(img, augment=False)
        # pred = model(img, augment=False)[0]

        # Using confidence threshold, eliminate low-confidence predictions
        pred = box_filter(pred, conf_thres=opt.conf_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #print('detections: ', det, '\n')
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            print(f'Processing image {count}/{dataset.nf}')  # nf is number of files

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            (Path(str(save_dir / 'labels'))).mkdir(parents=True, exist_ok=True) 
            s += '%gx%g ' % img.shape[2:]  # print string

            # detection loop where poses are calculated
            if len(det):
                det = det.cpu()

                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :18], shapes[0], shapes[1])  # native-space pred
                prediction_confidence = det[i, 18]

                # box_pred = det.clone().cpu()
                box_predn = det[0, :18].clone().cpu()

                # Denormalize the corner predictions 
                corners2D_pr = np.array(np.reshape(box_predn, [9, 2]), dtype='float32')

                #print(f"corners3D:\n{corners3D}")

                obj_list, obj_dict, keypoints_array, cuboids_array = declare_objects()
                # desired object index
                desired_obj_idx = obj_list.index(desired_object)

                #print(f"cuboids_array:\n{cuboids_array[desired_obj_idx][0].T}")
                #print(f"keypoint_array:\n{keypoints_array}")

                # print('Corners  3D:\n',corners3D[:3, :])
                # print('Cuboids array:\n', cuboids_array[desired_obj_idx][0].T)

                # print('Corners  2D pr:\n', corners2D_pr)
                # print('keypoints 2d:\n',gt_pose_info[0]['keypoints_2d'])
                # print('projected_cuboid:\n',gt_pose_info[0]['projected_cuboid'])

                # print(f"keypoints_array:\n{keypoints_array[0][desired_obj_idx][0].T}")

                # Calculate rotation and tranlation in rodriquez format
                # R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]*1000), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))
                #in mm instead of original m
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), cuboids_array[desired_obj_idx][0].T), axis=1)), dtype='float32'), corners2D_pr, np.array(internal_calibration, dtype='float32'))
                # t_pr *= 1000  # Convert from meters to millimeters

                # T = np.array([
                #     [-1,0,0],
                #     [0,0,1],
                #     [0,1,0]
                # ])
                # print(f"R_pr:\n{R_pr}")
                # print(f"t_pr:\n{t_pr}")
                # exit()
                # print(f"R_gt:\n{R_gt}")
                # print(f"t_gt:\n{t_gt}")

                pose_mat = cv2.hconcat((R_pr, t_pr))
                euler_angles = cv2.decomposeProjectionMatrix(pose_mat)[6]
                predictions.append([det, euler_angles, t_pr])

                # Store the pose
                estimated_poses.append(pose_mat)
                # end time information and calculation
                t2 = time_synchronized()
                inference_time_ms = (t2 - t1) * 1000  # Convert to milliseconds
                fps = 1.0 / (t2 - t1)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                if save_img:
                    # convert bgr to rgb
                    local_img       = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

                    figsize         = (shapes[0][1]/96, shapes[0][0]/96)
                    fig             = plt.figure(frameon=False, figsize=figsize)
                    ax              = plt.Axes(fig, [0., 0., 1., 1.])

                    ax.set_axis_off()
                    fig.add_axes(ax)

                    ### Draw GT BB ###

                    # Find matching ground truth frame
                    current_frame = os.path.splitext(os.path.basename(path))[0]
                    current_frame_str = f"{current_frame}.json"

                    matching_gt = None
                    
                    # Find matching ground truth pose
                    for gt_pose in gt_pose_info:
                        if gt_pose['filename'] == current_frame_str:
                            matching_gt = gt_pose
                            break
                    
                    if matching_gt is None:
                        print(f"No matching ground truth found for frame {current_frame_str}")
                        continue

                    local_img, current_gt_mat = draw_gt_pose(local_img, matching_gt, cuboids_array[desired_obj_idx][0], internal_calibration)

                    gt_poses.append(current_gt_mat)

                    # Draw estimated BB
                    corn2D_pr= corners2D_pr[1:, :] # Remove the first point (center) for drawing
                    
                    # print("corners2D_pr:\n", corners2D_pr)

                    local_img = draw_bb(
                                        corn2D_pr,  # Estimated corners
                                        local_img,
                                        color_lines=(255, 255, 255),  # white color for estimated
                                        color_points=(0, 0, 255),  # blue points
                                        width=2,  # Thinner lines for estimated truth
                                        line_type=cv2.LINE_AA  # Anti-aliased line
                                    )
                    # Draw estimated axes
                    local_img = draw_axes(local_img, corners2D_pr, colors=[(255, 0, 0), (0, 255, 0), (0, 125, 255)], thickness=2)



                    # Draw estimated center
                    est_center  = np.mean(corn2D_pr, axis=0).astype(int)
                    cv2.circle(local_img, 
                    (int(est_center[0]), int(est_center[1])), 
                    radius=5,           # Larger radius for visibility
                    color=(255, 255, 0),  # Orange
                    thickness=-1)       # Filled circle

                    # Textbox on screen
                    #ax.text(min_x, min_y-10, f"Conf: {prediction_confidence.cpu().numpy():.3f}, Rot: {euler_angles}, Trans: {t_pr.flatten()}", style='italic', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
                    
                    ax.imshow(local_img, aspect='auto')

                    filename = f"{count:04d}.png"
                    file_path = os.path.join(images_dir, filename)
                    fig.savefig(file_path, dpi = 96, bbox_inches='tight', pad_inches=0)
                    # fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
                    plt.close()

                    count += 1

                # Log timing information for this frame
                with open(timing_file, 'a') as f:
                    f.write("| {:6d} | {:12.2f} | {:7.2f} |\n".format(
                        count, 
                        inference_time_ms, 
                        fps
                    ))

                print(f'Done. Inference time: {inference_time_ms:.3f} ms')

                # Save results (image with detections)
                with open(txt_path + '.txt', 'a') as f:
                    f.write(f"Confidence Value:\n {prediction_confidence.cpu().numpy():.3f}\n")
                    f.write(f"Estimated Rotation:\n {R_pr}\n")
                    f.write(f"Estimated Position:\n {t_pr}\n")
                    f.write(f"2D Estimation:\n {corners2D_pr[0]}\n")
                    f.write(f"Estimated 6D Pose:\n {pose_mat}\n")
                    f.write(f"Estimated 2D Corners:\n {corn2D_pr}\n")

                inference_times.append(inference_time_ms)
            else:
                print(f"No detections found in {p}")
                failed_images.append(count)
                count += 1

                

    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    #print(f"Results saved to {save_dir}{s}")

    #print(f'Done. total processing time: {time.time() - t0:.3f}s')

    print(f"Failed images: {failed_images}")

    total_time = sum(inference_times)
    avg_time = total_time / len(inference_times)
    avg_fps = len(inference_times) / (total_time / 1000)  # Convert ms to seconds
    fps_values = [1000 / t for t in inference_times]  # Convert times to FPS

    with open(timing_file, 'a') as f:
        f.write("\n")
        f.write("=" * 50 + "\n")
        f.write("Inference Speed Summary:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total frames processed: {len(inference_times)}\n")
        f.write(f"Total inference time: {total_time/1000:.2f} seconds\n")
        f.write(f"Average inference time: {avg_time:.2f} ms\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Min FPS: {min(fps_values):.2f}\n")
        f.write(f"Max FPS: {max(fps_values):.2f}\n")

    print("\nComputing pose errors...")
    avg_trans_error, avg_rot_error, avg_rot_error_mat, avg_add_error = compute_pose_errors(
        estimated_poses,
        gt_poses,  # GT poses arr
        save_dir,  # Output path
        desired_object,  # Object ID
        failed_images,
        internal_calibration,  # Camera intrinsics
    )
    print(f"Average translation error: {avg_trans_error:.2f} mm")
    print(f"Average rotation error euler: {avg_rot_error:.2f} degrees")
    print(f"Average rotation error matrix: {avg_rot_error_mat:.2f} degrees")
    print(f"Average ADD error: {avg_add_error:.2f}\n")

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz_proj = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz_proj, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, xyz_proj

def draw_axes(img, keypoints, colors = [(255, 0, 0), (0, 255, 0),(0, 125, 255)], thickness = 2 ):
    # args: image, projected_cuboid keypoints, list of 3 colors to use, tickenss
    # returns the image with the line drawn

    """Draw 3D axes on the image.
    Keypoints ordering:
    0: center
    1-4: front-top-right, front-top-left, back-top-right, back-top-left
    5-8: front-bottom-right, front-bottom-left, back-bottom-right, back-bottom-left
    """

    # finds the center point
    center = np.mean(keypoints, axis=0)
    center = [int(i) for i in center]
    
    # finds the top of the object    
    # point1_top = [(keypoints[1][0] + keypoints[2][0])/2, (keypoints[1][1] + keypoints[2][1])/2]
    # point2_top = [(keypoints[0][0] + keypoints[3][0])/2, (keypoints[0][1] + keypoints[3][1])/2]
    # top_coords = [int((point1_top[0] + point2_top[0])/2), int((point1_top[1] + point2_top[1])/2)]
    
    # # finds the right of the top of the object
    # point1_right = [(keypoints[3][0] + keypoints[6][0])/2, (keypoints[3][1] + keypoints[6][1])/2]
    # point2_right = [(keypoints[2][0] + keypoints[7][0])/2, (keypoints[2][1] + keypoints[7][1])/2]
    # right_coords = [int((point1_right[0] + point2_right[0])/2), int((point1_right[1] + point2_right[1])/2)]
    
    # # finds the center of the front of the object
    # point1_front = [(keypoints[1][0] + keypoints[7][0])/2, (keypoints[1][1] + keypoints[7][1])/2]
    # point2_front = [(keypoints[3][0] + keypoints[5][0])/2, (keypoints[3][1] + keypoints[5][1])/2]
    # front_coords = [int((point1_front[0] + point2_front[0])/2), int((point1_front[1] + point2_front[1])/2)]

    # X-axis (red) - pointing up
    top_points = keypoints[1:5]  # Get all top points
    top_center = np.mean(top_points, axis=0)
    top_coords = [int(top_center[0]), int(top_center[1])]

    # Y-axis (green) - pointing right
    right_points = np.array([keypoints[1], keypoints[3], keypoints[5], keypoints[7]])  # All right side points
    right_center = np.mean(right_points, axis=0)
    right_coords = [int(right_center[0]), int(right_center[1])]
    
    # Z-axis (blue) - pointing front
    front_points = np.array([keypoints[1], keypoints[2], keypoints[5], keypoints[6]])  # All front points
    front_center = np.mean(front_points, axis=0)
    front_coords = [int(front_center[0]), int(front_center[1])]
    
    # draws lines
    cv2.line(img, center, top_coords, colors[0], thickness)
    cv2.putText(img, 'X', top_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)

    cv2.line(img, center, right_coords, colors[1], thickness)
    cv2.putText(img, 'Y', right_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2)

    cv2.line(img, center, front_coords, colors[2], thickness)
    cv2.putText(img, 'Z', front_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 2)

    return img

def read_obj8_poses(folder_path, desired_object):
    """
    Read 6D poses of object 8 from JSON files in the given folder.
    
    Args:
        folder_path: Path to folder containing JSON files
    
    Returns:
        list of dict: Each containing location and orientation for object 8
    """
    poses = []
    
    # List and sort all JSON files
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Find object 8 in the objects list
            for obj in data['objects']:
                if obj['class'] == desired_object:
                    # Get rotation and translation
                    pose_transform = np.array(obj['pose_transform'])
                    rotation = pose_transform[:3, :3]
                    translation = np.array(obj['location'])
                    projected_cuboid_lst = np.array(obj['projected_cuboid'])
                    keypoints_2d_lst = np.array(obj['keypoints_2d'])
                    
                    # Construct correct transformation matrix
                    correct_transform = np.eye(4)
                    correct_transform[:3, :3] = rotation
                    correct_transform[:3, 3] = translation
                    
                    pose_info = {
                        'filename': filename,
                        'location': translation,  # [x, y, z] in mm
                        'rotation_matrix': rotation.tolist(),  # 3x3 rotation matrix
                        'full_transform': correct_transform,  # 4x4 transformation matrix
                        'projected_cuboid': projected_cuboid_lst, # 8x2 projected cuboid points
                        'keypoints_2d': keypoints_2d_lst  # 8x2 keypoints
                    }
                    poses.append(pose_info)
                    break
    
    return poses

def declare_objects():
    """Returns object list, dictionary, keypoints, and cuboids."""

    obj_list = [
        'obj_000001', 'obj_000005', 'obj_000006', 'obj_000008',
        'obj_000009', 'obj_000010', 'obj_000011', 'obj_000012'
    ]

    obj_dict = {
        "obj_000001": "ape",
        "obj_000005": "wateringcan",
        "obj_000006": "cat",
        "obj_000008": "drill",
        "obj_000009": "duck",
        "obj_000010": "eggbox",
        "obj_000011": "glue",
        "obj_000012": "holepuncher"
    }
    keypoints_array = np.array([[
                                #"obj_000001": "ape"
                                [[[-9.7732497e-03, 3.6659201e-03, -1.4534000e-03],
                                [3.6046398e+01, -1.4680500e+01, -4.5020599e+01],
                                [-3.0289101e+01, -7.2402501e+00, -4.2632900e+01],
                                [7.4425201e+00, 2.3966400e+01, -3.9362701e+01],
                                [-4.3485899e+00, 3.6769500e+00, 4.5835800e+01],
                                [6.2882501e-01, -3.6412800e+01, -2.7732599e+01],
                                [-2.6754901e-01, 3.7588799e+01, -4.7640200e+00],
                                [3.0029400e+01, -2.3939800e+01, -8.1097898e+00],
                                [-2.8789900e+01, -1.9449200e+01, -9.0417604e+00]]],

                                #"obj_000005": "wateringcan"
                                [[[1.6300200e-02, -2.3040799e-03, -1.1291500e-02],
                                [5.5248199e+00, 5.4157101e+01, -9.6322701e+01],
                                [-4.1018100e+00, 1.2732400e+01, 9.6678497e+01],
                                [-9.1580000e+00, -4.1244202e+01, -8.7472397e+01],
                                [7.3375401e+00, 9.0886101e+01, -1.1365300e+01],
                                [-1.0262200e+01, -9.0547600e+01, -3.7563899e-01],
                                [-4.7794201e+01, 1.6508699e+01, -5.6376900e+01],
                                [4.8287998e+01, 2.4022501e+00, -6.2877899e+01],
                                [4.6154099e+01, 1.1302400e+01, 4.9851101e+01]]],

                                # "obj_000006": "cat"
                                [[[1.7128000e-02, -4.5700101e-03, -5.3901700e-03],
                                [2.0947300e+01, -6.1587502e+01, -5.4198200e+01],
                                [-2.0933701e+01, 6.3563000e+01, 2.6130899e+01],
                                [2.8901501e+01, 2.7392700e+01, -5.7568199e+01],
                                [1.4403200e+00, -5.8665901e+01, 2.2473900e+01],
                                [1.2946500e+01, 1.4082400e+01, 5.8292999e+01],
                                [-2.8743299e+01, 1.6301001e+01, -5.2558300e+01],
                                [-3.3441200e+01, -4.1310501e+01, -5.4232101e+01],
                                [2.3869900e+01, 4.1699699e+01, 1.6587299e+01]]],

                                #"obj_000008": "drill"
                                [[[-2.4108901e-03, -6.2332200e-03, -6.3247699e-03],
                                [1.1291000e+02, -3.4727199e+00, 9.2172699e+01],
                                [-1.1182900e+02, 3.1709600e-02, 6.1154400e+01],
                                [-6.2377201e+01, 1.0970700e+01, -1.0025700e+02],
                                [4.2661201e+01, -2.4666700e+01, -9.9452499e+01],
                                [1.0724100e+01, -3.5357201e+00, 1.0133300e+02],
                                [-4.1970699e+01, -3.1155399e+01, 5.4645599e+01],
                                [4.9310899e+00, 3.6434399e+01, -9.7123596e+01],
                                [5.6840302e+01, -4.2665200e+00, 4.8058399e+01]]],

                                #"obj_000009": "duck"
                                [[[-3.4179699e-03, -9.8838797e-03, 3.9329501e-03],
                                [4.9320702e+01, 6.2302999e+00, -4.0302898e+01],
                                [-4.6246700e+01, 2.3396499e+00, -3.7502899e+01],
                                [1.2448000e+01, -3.3365299e+01, -4.0734501e+01],
                                [3.9640200e+00, 3.4297600e+01, -4.0923302e+01],
                                [4.5272598e+01, -1.0067500e+00, 2.1399401e+01],
                                [6.6833901e+00, -3.1548400e+00, 4.2783199e+01],
                                [-2.3509399e+01, -2.7834400e+01, -1.9335600e+01],
                                [-4.1355202e+01, 1.3988900e-01, 1.3391900e+00]]],

                                #"obj_000010": "eggbox"
                                [[[-1.7417900e-02, -4.2999300e-01, -1.3252300e-02],
                                [-7.0443398e+01, 4.3526299e+01, 4.2999201e+00],
                                [7.3233902e+01, 3.5586300e+01, 4.8644700e+00],
                                [6.7131897e+01, -4.4466202e+01, -2.7725799e+00],
                                [-7.0990898e+01, -3.6974701e+01, -1.3353300e+00],
                                [-4.7924999e+01, 5.5036702e+00, -3.2836399e+01],
                                [2.2584101e+01, 4.1242500e+01, 3.2724400e+01],
                                [-2.4753901e+01, -4.0470100e+01, 3.2213699e+01],
                                [4.7744598e+01, 4.2735401e-01, -3.1653799e+01]]],

                                #"obj_000011": "glue"
                                [[[9.9391900e-03, -1.1459400e-02, 6.8359398e-03],
                                [-9.1147299e+00, -3.1402399e+01, -8.5777802e+01],
                                [9.7676700e-01, 2.9348700e+00, 8.6390404e+01],
                                [6.4356799e+00, 3.7870701e+01, -6.3978802e+01],
                                [9.7071304e+00, -3.6640800e+01, -3.6885799e+01],
                                [-1.5302700e+01, 1.4431200e+00, -4.7971500e+01],
                                [-6.0784298e-01, -1.2160700e+01, 4.3689098e+01],
                                [1.7079800e+01, 1.9666600e+00, -8.3763802e+01],
                                [-4.1084499e+00, 3.5197800e+01, -2.3239799e+01]]],
                                
                                #"obj_000012": "holepuncher"
                                [[[-0.00317764, -0.00389862, 0.0116768],
                                [45.3747, 47.1416, -37.7327],
                                [-34.1938, -52.1613, -38.8932],
                                [-37.2968, 48.9418, -39.6015],
                                [39.7582, 48.9375, 35.2916],
                                [44.2335, -42.2343, -37.7516],
                                [39.6273, -44.8461, 35.1788],
                                [-15.6101, 52.0532, 11.4769],
                                [-50.3098, -12.8502, -5.48413]]]
                                
                                ]])

    cuboids_array = np.array([
                                #"obj_000001": "ape"
                                [[[-37.92094, -38.788555, -45.88129],
                                [-37.92094, -38.788555, 45.87838],
                                [-37.92094, 38.795883, -45.88129],
                                [-37.92094, 38.795883, 45.87838],
                                [37.901394, -38.788555, -45.88129],
                                [37.901394, -38.788555, 45.87838],
                                [37.901394, 38.795883, -45.88129],
                                [37.901394, 38.795883, 45.87838]]],

                                #"obj_000005": "wateringcan"
                                [[[-50.35713, -90.89071, -96.8516],
                                [-50.35713, -90.89071, 96.82902],
                                [-50.35713, 90.8861, -96.8516],
                                [-50.35713, 90.8861, 96.82902],
                                [50.38973, -90.89071, -96.8516],
                                [50.38973, -90.89071, 96.82902],
                                [50.38973, 90.8861, -96.8516],
                                [50.38973, 90.8861, 96.82902]]],

                                # "obj_000006": "cat"
                                [[[-33.44303, -63.791, -58.71809],
                                [-33.44303, -63.791, 58.707314],
                                [-33.44303, 63.781857, -58.71809],
                                [-33.44303, 63.781857, 58.707314],
                                [33.47729, -63.791, -58.71809],
                                [33.47729, -63.791, 58.707314],
                                [33.47729, 63.781857, -58.71809],
                                [33.47729, 63.781857, 58.707314]]],

                                #"obj_000008": "drill"
                                [[[-114.72308, -37.718895, -103.983604],
                                [-114.72308, -37.718895, 103.97095],
                                [-114.72308, 37.706425, -103.983604],
                                [-114.72308, 37.706425, 103.97095],
                                [114.71827, -37.718895, -103.983604],
                                [114.71827, -37.718895, 103.97095],
                                [114.71827, 37.706425, -103.983604],
                                [114.71827, 37.706425, 103.97095]]],

                                #"obj_000009": "duck"
                                [[[-52.200897, -38.71081, -42.8214],
                                [-52.200897, -38.71081, 42.82927],
                                [-52.200897, 38.691044, -42.8214],
                                [-52.200897, 38.691044, 42.82927],
                                [52.194057, -38.71081, -42.8214],
                                [52.194057, -38.71081, 42.82927],
                                [52.194057, 38.691044, -42.8214],
                                [52.194057, 38.691044, 42.82927]]],

                                #"obj_000010": "eggbox"
                                [[[-75.0917, -54.39756, -34.629425],
                                [-75.0917, -54.39756, 34.602924],
                                [-75.0917, 53.53758, -34.629425],
                                [-75.0917, 53.53758, 34.602924],
                                [75.05686, -54.39756, -34.629425],
                                [75.05686, -54.39756, 34.602924],
                                [75.05686, 53.53758, -34.629425],
                                [75.05686, 53.53758, 34.602924]]],

                                #"obj_000011": "glue"
                                [[[-18.320473, -38.923126, -86.376724],
                                [-18.320473, -38.923126, 86.3904],
                                [-18.320473, 38.900208, -86.376724],
                                [-18.320473, 38.900208, 86.3904],
                                [18.340351, -38.923126, -86.376724],
                                [18.340351, -38.923126, 86.3904],
                                [18.340351, 38.900208, -86.376724],
                                [18.340351, 38.900208, 86.3904]]],

                                #"obj_000012": "holepuncher"
                                [[[-50.4439, -54.2485, -45.4],
                                [-50.4439, -54.2485, 45.4],
                                [-50.4439, 54.2485, -45.4],
                                [-50.4439, 54.2485, 45.4],
                                [50.4440, -54.2485, -45.4],
                                [50.4440, -54.2485, 45.4],
                                [50.4440, 54.2485, -45.4],
                                [50.4440, 54.2485, 45.4]]]
                                
                                ])
    
    return obj_list, obj_dict, keypoints_array, cuboids_array

def compute_pose_errors(est_poses, gt_poses, output_path, desired_object, failed_images, K):
    """
    Compare estimated poses with ground truth and compute errors.
    
    Args:
        est_poses: List of estimated poses from inference
        gt_folder_path: Path to ground truth JSON files
        output_path: Path to save error analysis
        obj_list: List of object IDs
    """
    obj_list, obj_dict, keypoints_array, cuboids_array = declare_objects()
    # Get ground truth poses
    #gt_poses = read_obj8_poses(gt_folder_path, desired_object)
    obj_idx = obj_list.index(desired_object)
    _, _, keypoints_array, cuboids_array = declare_objects()
    model_points = keypoints_array[0][obj_idx][0]  # Skip center point, use 8 keypoints
    model_points_cuboid = cuboids_array[obj_idx][0]

    model_diameter = np.max(pdist(model_points_cuboid))

    threshold = 0.1 * model_diameter  # 10% of diameter

    print("\nModel Diameter: ", model_diameter)
    print("Threshold: ", threshold,'\n')
    
    translation_errors = []
    rotation_errors = []
    rotation_errors_mat =[]
    failed_frames = []
    add_errors = []
    correct_pose_count = 0
    total_poses = 0
    
    # Create error log file
    error_file = os.path.join(output_path, "pose_errors.txt")
    with open(error_file, "w") as f:
        f.write(f"Pose Errors for {desired_object}\n")
        f.write("=" * 50 + "\n\n")
        
        # Process each frame
        for frame_idx, (est_pose, gt_pose_mat) in enumerate(zip(est_poses, gt_poses)):
            try:
                # Get estimated pose components
                est_R = est_pose[:3, :3]
                est_t = est_pose[:3, 3]

                # transforms to align the estimated pose with the ground truth pose axes
                T_axis = np.array([
                    [-1,0,0],
                    [0,-1,0],
                    [0,0,1]
                ]) 

                # no need to transform for holepuncher, duck
                est_R = est_R @ T_axis 
                #est_t = np.dot(T, est_t)

                # Store transformed estimated pose
                transformed_est_pose = np.zeros((3, 4))
                transformed_est_pose[:3, :3] = est_R
                transformed_est_pose[:3, 3] = est_t

                # Get ground truth components
                gt_R = gt_pose_mat[:3, :3]
                gt_t = gt_pose_mat[:3, 3]

                # print("Estimated Pose:\n",transformed_est_pose)
                # print("Ground Truth mat:\n",gt_pose_mat)
                # print("Ground Truth Rotation:\n",gt_R)

                # Compute translation error vector and magnitude (in mm)
                trans_error_vector = est_t - gt_t
                trans_error = np.linalg.norm(trans_error_vector)  # Magnitude
                
                # Convert rotation matrices to Euler angles (in degrees)
                est_euler = Rotation.from_matrix(est_R).as_euler('xyz', degrees=True)
                gt_euler = Rotation.from_matrix(gt_R).as_euler('xyz', degrees=True)

                # Compute rotation errors for each axis
                rot_error_vector = est_euler - gt_euler
                rot_error = np.linalg.norm(rot_error_vector)  # Total rotation error

                # Calculate rotation error using matrix logarithm (better than Euler angles)
                R_diff = est_R @ gt_R.T
                val = (np.trace(R_diff) - 1) / 2
                val_clamped = np.clip(val, -1.0, 1.0)
                rot_error_mat = np.rad2deg(np.arccos(val_clamped))

                # Calculate ADD error

                transformed_est = np.dot(model_points_cuboid, est_R.T) + est_t
                transformed_gt = np.dot(model_points_cuboid, gt_R.T) + gt_t
                # Calculate point-wise distances
                distances = np.linalg.norm(transformed_est - transformed_gt, axis=1)
                add_error = np.mean(distances)
                add_errors.append(add_error)

                gt_keypoints_2d, gt_keypoints_3d = project(model_points_cuboid, K, gt_pose_mat)
                est_keypoints_2d, est_keypoints_3d = project(model_points_cuboid, K, transformed_est_pose)


                # print(f"\nTransformed Estimated Points:\n{transformed_est}")
                # print(f"\nTransformed Estimated Points 3D:\n{est_keypoints_3d}")
                # print(f"\nTransformed Estimated Points 2D:\n{est_keypoints_2d}")
                # print(f"\nTransformed GT Points:\n{transformed_gt}")
                # print(f"\nTransformed GT Points 3D:\n{gt_keypoints_3d}")
                # print(f"\nTransformed GT Points 2D:\n{gt_keypoints_2d}")
                
                # Store errors
                translation_errors.append(trans_error)
                rotation_errors.append(rot_error)
                if not np.isnan(rot_error_mat):
                    rotation_errors_mat.append(rot_error_mat)

                # compute if the pose is correct
                is_correct_pose = np.floor(add_error) <= np.ceil(threshold)
                if is_correct_pose:
                    correct_pose_count += 1
                total_poses += 1
                
                # Write frame results
                f.write(f"Frame {frame_idx}:\n")
                f.write("\nEstimated Pose:\n")
                f.write("Rotation:\n")
                f.write(f"{est_R[0]}\n{est_R[1]}\n{est_R[2]}\n")
                f.write(f"Roll (X): {est_euler[0]:.2f}\n")
                f.write(f"Pitch (Y): {est_euler[1]:.2f}\n")
                f.write(f"Yaw (Z): {est_euler[2]:.2f}\n")
                f.write("Translation: ")
                f.write(f"[{est_t[0]:.2f}, {est_t[1]:.2f}, {est_t[2]:.2f}]\n")

                f.write("\nGround Truth Pose:\n")
                f.write("Rotation:\n")
                f.write(f"{gt_R[0]}\n{gt_R[1]}\n{gt_R[2]}\n")
                f.write("Euler Angles (xyz, degrees):\n")
                f.write(f"Roll (X): {gt_euler[0]:.2f}\n")
                f.write(f"Pitch (Y): {gt_euler[1]:.2f}\n")
                f.write(f"Yaw (Z): {gt_euler[2]:.2f}\n")
                f.write("Translation: ")
                f.write(f"[{gt_t[0]:.2f}, {gt_t[1]:.2f}, {gt_t[2]:.2f}]\n")

                # Write error analysis
                f.write("\nError Analysis:\n")
                f.write("Rotation Error (degrees):\n")
                f.write(f"Roll (X): {rot_error_vector[0]:.2f}\n")
                f.write(f"Pitch (Y): {rot_error_vector[1]:.2f}\n")
                f.write(f"Yaw (Z): {rot_error_vector[2]:.2f}\n")
                f.write(f"Total Rotation Error Euler: {rot_error:.2f}\n")
                f.write(f"Total Rotation Error Matrix: {rot_error_mat:.2f}\n")
                f.write("Translation Error Vector [x, y, z]: ")
                f.write(f"[{trans_error_vector[0]:.2f}, {trans_error_vector[1]:.2f}, {trans_error_vector[2]:.2f}] mm\n")
                f.write(f"Translation Error Magnitude: {trans_error:.2f} mm\n")

                f.write("\nADD Error Analysis:\n")
                f.write(f"ADD Error: {add_error:.2f}\n")
                f.write(f"Point-wise distances: {distances}\n")
                f.write(f"Model diameter: {model_diameter:.2f} mm\n")
                f.write(f"10% threshold: {threshold:.2f} mm\n")
                f.write(f"Pose correct: {is_correct_pose}\n")
                f.write(f"Percentage of model diameter: {add_error/model_diameter*100:.2f}\n")
                
                f.write("-" * 30 + "\n")

                print(f"Frame {frame_idx} ADD Error: {add_error}")
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                failed_frames.append(frame_idx)
                continue
        
        if translation_errors:
            # Compute statistics
            avg_trans_error = np.mean(translation_errors)
            avg_rot_error = np.mean(rotation_errors)
            avg_rot_error_mat = np.mean(rotation_errors_mat)
            avg_add_error = np.mean(add_errors)
            
            # Standard deviation
            std_trans_error = np.std(translation_errors)
            std_rot_error = np.std(rotation_errors)
            std_rot_error_mat = np.std(rotation_errors_mat)
            std_add_error = np.std(add_errors)

            # Write summary statistics
            f.write("\nSummary Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of frames processed: {len(translation_errors)}\n")
            f.write(f"Number of failed frames: {len(failed_images)}\n")
            f.write(f"Failed frames: {failed_images}\n")
            f.write(f"Average translation error: {avg_trans_error:.2f} +- {std_trans_error:.2f} mm\n")
            f.write(f"Average rotation error euler: {avg_rot_error:.2f} +- {std_rot_error:.2f} degrees\n")
            f.write(f"Average rotation error mat: {avg_rot_error_mat:.2f} +- {std_rot_error_mat:.2f} degrees\n")
            f.write(f"Average ADD error: {avg_add_error:.2f} +- {std_add_error:.2f}\n")
            f.write(f"Number of correct poses: {correct_pose_count}\n")
            f.write(f"Number of ADD errors > 100: {np.sum(np.array(add_errors) >= 100)}\n")
            f.write(f"ADD error array: {add_errors}\n")

            print("\nPose Error Statistics:")
            print(f"Number of frames processed: {len(translation_errors)}\n")
            print(f"Number of failed frames: {len(failed_frames)}\n")
            print(f"Number of correct poses: {correct_pose_count}")
            print(f"Number of ADD errors > 100: {np.sum(np.array(add_errors) >= 100)}")

            return avg_trans_error, avg_rot_error, avg_rot_error_mat, avg_add_error
        else:
            print("No valid poses found for comparison")
            return float('nan'), float('nan')

def draw_gt_pose(img, gt_pose_info, model_points, K):
    """Draw ground truth pose on image."""
    
    # Extract rotation and translation from full transform
    gt_R = np.array(gt_pose_info['rotation_matrix']).T
    gt_t = gt_pose_info['location']
    
    # Apply transformation T to rotation matrix
    #gt_R = np.dot(T, gt_R)

    # Reconstruct transformation matrix
    gt_mat = np.zeros((3, 4))
    gt_mat[:3, :3] = gt_R
    gt_mat[:3, 3] = gt_t
    
    # Project 3D points to 2D
    gt_keypoints_2d, gt_keypoints_3d = project(model_points, K, gt_mat)
    # Convert to integer coordinates
    gt_keypoints_2d = gt_keypoints_2d.astype(np.int32)

    # print(f"GT Keypoints 3D:\n{gt_keypoints_3d}")
    # print(f"GT Keypoints 2D:\n{gt_keypoints_2d}")
    # print(f"Ground Truth Mat:\n{gt_mat}")
    # print(f"Ground Truth Rotation:\n{gt_R}")

    # Calculate center of the object in 2D
    center = np.mean(gt_keypoints_2d, axis=0).astype(int)
    # cv2.circle(img, 
    #             tuple(center),
    #             radius=5,           # Larger radius for visibility
    #             color=(255, 255, 0),  # Yellow
    #             thickness=-1)       # Filled circle

    img = draw_bb(
        gt_keypoints_2d,
        img,
        color_lines=(0, 255, 255),  # aqua color for ground truth
        color_points=(0, 255, 0),  # yellow points
        width=1,  # Thinner lines for ground truth
        line_type=cv2.LINE_AA  # Anti-aliased line
    )

    return img, gt_mat

def draw_bb(
    xy,
    img,
    color_lines=(0, 0, 255),
    color_points=(0, 255, 255),
    width=2,
    line_type=cv2.LINE_4,
):
    xy = xy.astype(int)
    xy = tuple(map(tuple, xy))
    cv2.line(img, xy[0], xy[1], color_lines, width, line_type)
    cv2.line(img, xy[1], xy[3], color_lines, width, line_type)
    cv2.line(img, xy[3], xy[2], color_lines, width, line_type)
    cv2.line(img, xy[2], xy[0], color_lines, width, line_type)
    cv2.line(img, xy[0], xy[4], color_lines, width, line_type)
    cv2.line(img, xy[4], xy[5], color_lines, width, line_type)
    cv2.line(img, xy[5], xy[7], color_lines, width, line_type)
    cv2.line(img, xy[7], xy[6], color_lines, width, line_type)
    cv2.line(img, xy[6], xy[4], color_lines, width, line_type)
    cv2.line(img, xy[2], xy[6], color_lines, width, line_type)
    cv2.line(img, xy[7], xy[3], color_lines, width, line_type)
    cv2.line(img, xy[1], xy[5], color_lines, width, line_type)

    for p in xy:
        cv2.circle(img, p, 1, color_points, -1)
    return img

if __name__ == '__main__':
    '''
    Example usage:
    python detect.py --weights ./weights/LINEMOD/driller/weights/best.pt
                    --img 640 --conf 0.15 --source ./data/lmo_report_test/ten_img/rgb
                    --static-camera configs/linemod/linemod_camera.json
                    --mesh-data ./data/LINEMOD/driller/driller.ply
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l6_pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--static-camera', type=str, help='path to static camera intrinsics')
    parser.add_argument('--mesh-data', type=str, help='path to object specific mesh data')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    opt = parser.parse_args()
    #print(opt)

    with torch.no_grad():
        detect()

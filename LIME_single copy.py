#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:27:59 2021

@author: 
"""

import argparse
import numpy as np
import os
import torch
import logging
import sys
import importlib
# from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import open3d as o3d
import time

from lime import lime_3d_remove

# sys.path.append('/home/as_admin/development/LIME-3D/models')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GNET_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))  # Adding required pointnet2 modules to system path
sys.path.append(os.path.join(GNET_DIR, 'models'))  # GrapNet

from graspnet import GraspNet

# SHAPE_NAMES = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 
SHAPE_NAMES = [line.rstrip().lstrip() for line in open(os.path.join(BASE_DIR, 'data/shape_names.txt')).readlines()]

def take_second(elem):
    return elem[1]

def take_first(elem):
    return elem[0]

def gen_pc_data(ori_data,segments,explain,label,filename):
    """
    NOTE:
    Args:
        ori_data: n (usually 1024) sampled piontcloud points
        segments: int array of length 1024
        explain: dict{key: [tuple, ], } each tuple is a pair of (int, float) and each value in dict is a 20 long list of these
        l: int corresponding to the id of the label?
        filename...
    Returns:
        colourful pc
    """
    basic_path = os.path.join(BASE_DIR, 'visu')
    color = np.zeros([ori_data.shape[0],3])
    max_contri = 0
    min_contri = 0
    for k in explain[label]:
        if k[1] > 0 and k[1] > max_contri:
            max_contri = k[1]
        elif k[1] < 0 and k[1] < min_contri:
            min_contri = k[1]
    if max_contri > 0:
        positive_color_scale = 1/max_contri
    
    else:
        positive_color_scale = 0
    if min_contri < 0:
        negative_color_scale = 1/min_contri
    else:
        negative_color_scale = 0
    ex_sorted = sorted(explain[label],key=take_first,reverse=False)
    for i in range(segments.shape[0]):
        if ex_sorted[segments[i]][1] > 0:
            color[i][0] = ex_sorted[segments[i]][1] * positive_color_scale
        elif ex_sorted[segments[i]][1] < 0:
            color[i][2] = ex_sorted[segments[i]][1] * negative_color_scale
        else:
            color[i] = [0,0,0]
    pc_colored = np.concatenate((ori_data,color),axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_colored[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(pc_colored[:,3:6])
    o3d.io.write_point_cloud(os.path.join(basic_path, filename), pc)        
    # print("Generate point cloud (gen pc data)", filename, "successful!") 
    return
    
def reverse_points(points,segments,explain,start='positive',percentage=0.2):
    num_input_dims = points.shape[1]
    basic_path = os.path.join(BASE_DIR, 'output')
    filename = 'reversed.ply'
    if start == 'positive':
        to_rev_list = np.argsort(explain)[-int(len(explain)*percentage):]
        to_rev_list = to_rev_list[::-1]
    elif start == 'negative':
        to_rev_list = np.argsort(explain)[:int(len(explain)*percentage)]
    else:
        print('Wrong start input!')
        return points
    for i in range(len(to_rev_list)):
        segment_to_rev = to_rev_list[i]
        rev_points_index = np.argwhere(segments==segment_to_rev)
        for p in rev_points_index:
            points[p] = np.zeros([num_input_dims])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:,0:3])
    o3d.io.write_point_cloud(os.path.join(basic_path, filename), pc)
    # print("Generate point cloud (reverse points)", filename, "successful!") 
    return points

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='cuda', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='log/logs', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--model_name', type=str, default='pointnet2_cls_ssg')
    parser.add_argument('--normals', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='log/checkpoints/pointnet2_cls_ssg.pth')
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--input_file_path', type=str, default='/home/as_admin/development/LIME-3D/data/ModelNet40_converted/airplane/airplane_0001.ply')
    return parser.parse_args()

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

# GPT test function
# def test(model, loader, num_class=40, vote_num=1):
#     if loader[-3:] == 'npy':
#         points = np.load(loader)
#     elif loader[-3:] == 'txt':
#         points = np.loadtxt(loader, delimiter=',')
#     elif loader[-3:] == 'ply':
#         pc = o3d.io.read_point_cloud(loader)

#         Check if normals exist; if not, compute them
#         if not pc.has_normals():
#             print("⚠️ PLY file lacks normals. Computing them...")
#             pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        
#         points = np.hstack((np.asarray(pc.points), np.asarray(pc.normals)))  # ✅ Stack XYZ + Normals

#     # Ensure the model receives 6 channels
#     if points.shape[1] != 6:
#         raise ValueError(f"❌ Expected 6 channels (XYZ + Normals), but got {points.shape[1]}")

#     if points.shape[0] > 1024:
#         points = sampling(points, 1024)
    
#     points = np.expand_dims(points, 0)
#     points = torch.from_numpy(points).transpose(2, 1).float()

#     classifier = model.eval()
#     pred, bf_sftmx, _ = classifier(points)
#     pred_choice = pred.data.max(1)[1]

#     print('Fc3 layer score:\n', bf_sftmx[0][pred_choice])
#     print('Prediction Score:\n', pred[0])
#     print('Predict Result: ', pred_choice, SHAPE_NAMES[pred_choice])

#     return points, pred_choice, pred


def test(model, loader):
    if loader[-3:] == 'npy':
        points = np.load(loader)
    elif loader[-3:] == 'txt':
        points = np.loadtxt(loader,delimiter=',')
    elif loader[-3:] == 'ply':
        points = o3d.io.read_point_cloud(loader)
        points = np.asarray(points.points)
    if points.shape[1] > 3:
        points = points[:,0:3]
    if points.shape[0] > 1024:
        points = sampling(points,1024)
    elif points.shape[0] < 1024:
        print("PC TOO SMALL")
    points = np.expand_dims(points,0)
    points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.float()
    classifier = model.eval()
    pred, bf_sftmx = classifier(points)
    pred_choice = pred.data.max(1)[1]
    print('Fc3 layer score:\n', bf_sftmx[0][pred_choice])
    print('Prediction Score:\n', pred[0])
    print('Predict Result: ',pred_choice, SHAPE_NAMES[pred_choice])
    return points, pred_choice, pred

def show_pc(dir, filename):
    """
    Plots coloured PC explanations using matplotlib. 
    RED - more positive contribution to classification
    BLUE - more negative contribution
    DIM POINTS - 0 contribution
    """
    basic_path = os.path.join(BASE_DIR, dir, filename)
    pc = o3d.io.read_point_cloud(basic_path)

    # Convert to NumPy arrays
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors) if dir == 'visu' else None # Colors are normalized between 0 and 1

    # Create Matplotlib 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=3)
    plt.show()
    return

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' if args.gpu == 'cuda' else '1'

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % os.path.join(BASE_DIR, args.log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model_name)
    # GraspNet
    MODEL = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                     cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    # classifier = MODEL.get_model(args.num_classes, normal_channel=args.normals)
    classifier = MODEL
    # checkpoint = torch.load(os.path.join(BASE_DIR, args.checkpoint_path), map_location=torch.device(args.gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cfgs.checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    # classifier.load_state_dict(checkpoint)
    # filename = f'{BASE_DIR}/data/modelnet40_normal_resampled/wardrobe/wardrobe_0002.txt'
    filename = args.input_file_path
    show_pc('data/ModelNet40_converted/airplane', os.path.join(args.input_file_path.rsplit('/')[-1]))
    with torch.no_grad():
        points, pred, logits = test(classifier.eval(), filename)
        
    l = pred.detach().numpy()[0]  # NOTE: detatch removes the computational graph of the torch.tensor, then this is converted to a np array and we take the 1st element in this 
    points_for_exp = np.asarray(points.squeeze().transpose(1,0))
    predict_fn = classifier.eval()  # essentially the pointnet model
    explainer = lime_3d_remove.LimeImageExplainer(random_state=0)
    tmp = time.time()
    explanation = explainer.explain_instance(points_for_exp, predict_fn, top_labels=5, num_features=20, num_samples=10, random_seed=0)
    print ('Completed in: ',time.time() - tmp,'s')
    #temp, mask = explaination.get_image_and_mask(l, positive_only=False, negative_only=False, num_features=100, hide_rest=True)
    #gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,(str(start_idx)+'_'+SHAPE_NAMES[pred_val[i-start_idx]]+'_gt_is_'+SHAPE_NAMES[l]+'_'+'_lime.ply'))
    
    gen_pc_data(points_for_exp,explanation.segments,explanation.local_exp,l,args.input_file_path.rsplit('/')[-1])
    show_pc('visu', os.path.join(args.input_file_path.rsplit('/')[-1]))
    return explanation
    
if __name__ == '__main__':
    args = parse_args()
    exp = main(args)
    
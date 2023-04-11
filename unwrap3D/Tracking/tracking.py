# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:50:05 2022

@author: felix
"""

from ..Utility_Functions import file_io as fio
import numpy as np 


# is there a quick way to compute the occupied bbox density. 
def remove_very_large_bbox(boxes, shape, thresh=0.5, aspect_ratio=None, method='density', max_density=4, mode='fast', return_ind=False):
    
    from sklearn.metrics.pairwise import pairwise_distances
    
    if method == 'density':
        
        boxes_ = boxes.copy()
        keep_ind = np.arange(len(boxes_))
        
        if aspect_ratio is not None:
            w = boxes[:,2] - boxes[:,0]
            h = boxes[:,3] - boxes[:,1]
            wh = np.vstack([w,h])
            
            aspect_w_h = np.max(wh, axis=0) / (np.min(wh, axis=0) + .1)
            boxes_ = boxes_[aspect_w_h<=aspect_ratio]
            keep_ind = keep_ind[aspect_w_h<=aspect_ratio]
        
        box_density = np.zeros(shape)
        bbox_coverage = np.zeros(len(boxes_))
        box_centroids_x = np.clip((.5*(boxes_[:,0] + boxes_[:,2])).astype(np.int), 0, shape[1]-1).astype(np.int)
        box_centroids_y = np.clip((.5*(boxes_[:,1] + boxes_[:,3])).astype(np.int), 0, shape[0]-1).astype(np.int)
        
        if mode == 'fast':
            box_centroids = np.vstack([box_centroids_x, box_centroids_y]).T
            box_centroids_r = np.sqrt((boxes_[:,2]-boxes_[:,0])*(boxes_[:,3]-boxes_[:,1])/np.pi +.1)
        
            box_centroids_distance = pairwise_distances(box_centroids)
            bbox_coverage = np.nansum(box_centroids_distance<=box_centroids_r[:,None], axis=1)
            
        else:
            # box_density[box_centroids_y, box_centroids_x] += 1
            for cent_ii in np.arange(len(box_centroids_x)):
                cent_x = box_centroids_x[cent_ii]
                cent_y = box_centroids_y[cent_ii]
                box_density[int(cent_y), int(cent_x)] += 1
            
            for box_ii, box in enumerate(boxes_):
                x1, y1, x2, y2 = box
                x1 = np.clip(int(x1), 0, shape[1]-1)
                y1 = np.clip(int(y1), 0, shape[0]-1)
                x2 = np.clip(int(x2), 0, shape[1]-1)
                y2 = np.clip(int(y2), 0, shape[0]-1)
                bbox_coverage[box_ii] = np.nansum(box_density[int(y1):int(y2), 
                                                              int(x1):int(x2)])        
        # print(bbox_coverage)
        if return_ind == False:
            return boxes_[bbox_coverage<=max_density]
        else:
            return boxes_[bbox_coverage<=max_density], keep_ind[bbox_coverage<=max_density]

    if method == 'area':
        areas_box = []
        area_shape = float(np.prod(shape))
        
    #    print(area_shape)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            area_box = (y2-y1)*(x2-x1)
            areas_box.append(area_box)
            
        areas_box = np.hstack(areas_box)
        areas_box_frac = areas_box / float(area_shape)
        return boxes[areas_box_frac<=thresh]



def Eval_dense_optic_flow(prev, present, params):
    r""" Computes the optical flow using Farnebacks Method

    Parameters
    ----------
    prev : numpy array
        previous frame, m x n image
    present :  numpy array
        current frame, m x n image
    params : Python dict
        a dict object to pass all algorithm parameters. Fields are the same as that in the opencv documentation, https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html. Our recommended starting values:
                
            * params['pyr_scale'] = 0.5
            * params['levels'] = 3
            * params['winsize'] = 15
            * params['iterations'] = 3
            * params['poly_n'] = 5
            * params['poly_sigma'] = 1.2
            * params['flags'] = 0
        
    Returns
    -------
    flow : finds the displacement field between frames, prev and present such that :math:`\mathrm{prev}(y,x) = \mathrm{next}(y+\mathrm{flow}(y,x)[1], x+\mathrm{flow}(y,x)[0])` where (x,y) is the cartesian coordinates of the image.
    """
    
    import numpy as np 
    import warnings
    import cv2

    # Check version of opencv installed, if not 3.0.0 then issue alert.
#    if '3.0.0' in cv2.__version__ or '3.1.0' in cv2.__version__:
        # Make the image pixels into floats.
    prev = prev.astype(np.float)
    present = present.astype(np.float)

    if cv2.__version__.split('.')[0] == '3' or cv2.__version__.split('.')[0] == '4':
        flow = cv2.calcOpticalFlowFarneback(prev, present, None, params['pyr_scale'], params['levels'], params['winsize'], params['iterations'], params['poly_n'], params['poly_sigma'], params['flags']) 
    if cv2.__version__.split('.')[0] == '2':
        flow = cv2.calcOpticalFlowFarneback(prev, present, pyr_scale=params['pyr_scale'], levels=params['levels'], winsize=params['winsize'], iterations=params['iterations'], poly_n=params['poly_n'], poly_sigma=params['poly_sigma'], flags=params['flags']) 
#    print(flow.shape)
    return flow


def rescale_intensity_percent(img, intensity_range=[2,98]):

    from skimage.exposure import rescale_intensity
    import numpy as np 

    p2, p98 = np.percentile(img, intensity_range)
    img_ = rescale_intensity(img, in_range=(p2,p98))

    return img_


# add in optical flow. 
def extract_optflow(vid, optical_flow_params, rescale_intensity=True, intensity_range=[2,98]): 
    # uses CV2 built in farneback ver. which is very fast and good for very noisy and small motion
    import cv2
    from skimage.exposure import rescale_intensity
    import numpy as np
    from tqdm import tqdm 

    vid_flow = []
    n_frames = len(vid)

    for frame in tqdm(np.arange(len(vid)-1)):
        frame0 = rescale_intensity_percent(vid[frame], intensity_range=intensity_range)
        frame1 = rescale_intensity_percent(vid[frame+1], intensity_range=intensity_range)
        flow01 = Eval_dense_optic_flow(frame0, frame1, 
                                       params=optical_flow_params)
        vid_flow.append(flow01)
    vid_flow = np.array(vid_flow).astype(np.float32) # to save some space. 

    return vid_flow



def predict_new_boxes_flow_tf(boxes, flow):
    
    from skimage.transform import estimate_transform, matrix_transform, SimilarityTransform
    from skimage.measure import ransac
    import numpy as np 
    
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
      
    new_boxes_tf = []
    tfs = []
    
    for box in boxes:
        x1,y1,x2,y2 = box
        
        x1 = int(x1); y1 =int(y1); x2=int(x2); y2 =int(y2);  # added, weirdly ...
        
        # how to take into account the change in size (scale + translation ? very difficult. )
        flow_x_patch = flow_x[y1:y2, x1:x2].copy()
        flow_y_patch = flow_y[y1:y2, x1:x2].copy()
        
        nrows_, ncols_ = flow_x_patch.shape
        
        # threshold_the mag
        mag_patch = np.sqrt(flow_x_patch ** 2 + flow_y_patch ** 2)
        select = mag_patch > 0
#        select = np.ones(mag_patch.shape) > 0
        pix_X, pix_Y = np.meshgrid(range(ncols_), range(nrows_))
        
        if np.sum(select) == 0:
            # if we cannot record movement in the box just append the original ?
            tfs.append([])
            new_boxes_tf.append([x1,y1,x2,y2])
        else:
            src_x = pix_X[select].ravel(); dst_x = src_x + flow_x_patch[select]
            src_y = pix_Y[select].ravel(); dst_y = src_y + flow_y_patch[select]
            src = np.hstack([src_x[:,None], src_y[:,None]])
            dst = np.hstack([dst_x[:,None], dst_y[:,None]])
            
            # estimate the transformation. 
            matrix = estimate_transform('similarity', src[:,[0,1]], dst[:,[0,1]])
            tf_scale = matrix.scale; tf_offset = matrix.translation
            
#            print tf_scale, tf_offset
            if np.isnan(tf_scale):
                tfs.append(([]))
                new_boxes_tf.append([x1,y1,x2,y2])
            else:
                x = .5*(x1+x2); w = x2-x1
                y = .5*(y1+y2); h = y2-y1
                
                x1_tf_new = x + tf_offset[0] - w*tf_scale/2.
                y1_tf_new = y + tf_offset[1] - h*tf_scale/2.
                x2_tf_new = x1_tf_new + w*tf_scale
                y2_tf_new = y1_tf_new + h*tf_scale
                
        #        print x1_tf_new
                tfs.append(matrix)
                new_boxes_tf.append([x1_tf_new, y1_tf_new, x2_tf_new, y2_tf_new])
        
    new_boxes_tf = np.array(new_boxes_tf)
    
    return tfs, new_boxes_tf

# this version is just over the boxes. given voc format. 
def bbox_iou_corner_xy(bboxes1, bboxes2):
    
    import numpy as np 
    
    """
    computes the distance matrix between two sets of bounding boxes.
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = bboxes1[:,0], bboxes1[:,1], bboxes1[:,2], bboxes1[:,3]
    x21, y21, x22, y22 = bboxes2[:,0], bboxes2[:,1], bboxes2[:,2], bboxes2[:,3]


    x11 = x11[:,None]; y11 = y11[:,None]; x12=x12[:,None]; y12=y12[:,None]
    x21 = x21[:,None]; y21 = y21[:,None]; x22=x22[:,None]; y22=y22[:,None]

    xI1 = np.maximum(x11, np.transpose(x21))
    xI2 = np.minimum(x12, np.transpose(x22))

    yI1 = np.maximum(y11, np.transpose(y21))
    yI2 = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xI2 - xI1), 0.) * np.maximum((yI2 - yI1), 0.)

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union+0.0001)


def bbox_tracks_2_array(vid_bbox_tracks, nframes, ndim):

    import numpy as np 
    N_tracks = len(vid_bbox_tracks)

    vid_bbox_tracks_all_array = np.zeros((N_tracks, nframes, ndim))
    vid_bbox_tracks_all_array[:] = np.nan

    vid_bbox_tracks_prob_all_array = np.zeros((N_tracks, nframes))
    vid_bbox_tracks_prob_all_array[:] = np.nan

    for ii in np.arange(N_tracks):
        tra_ii = vid_bbox_tracks[ii]
        tra_ii_times = np.array([tra_iii[0] for tra_iii in tra_ii])
        tra_ii_boxes = np.array([tra_iii[-1] for tra_iii in tra_ii]) # boxes is the last array. 
        tra_ii_prob = np.array([tra_iii[1] for tra_iii in tra_ii])

        vid_bbox_tracks_all_array[ii,tra_ii_times] = tra_ii_boxes.copy()
        vid_bbox_tracks_prob_all_array[ii,tra_ii_times] = tra_ii_prob.copy()

    return vid_bbox_tracks_prob_all_array, vid_bbox_tracks_all_array


def bbox_scalar_2_array(vid_scalar_tracks, nframes):

    import numpy as np 
    N_tracks = len(vid_scalar_tracks)

    vid_scalar_tracks_all_array = np.zeros((N_tracks, nframes))
    vid_scalar_tracks_all_array[:] = np.nan

    for ii in np.arange(N_tracks):
        tra_ii = vid_scalar_tracks[ii]
        tra_ii_times = np.array([tra_iii[0] for tra_iii in tra_ii])
        tra_ii_vals = np.array([tra_iii[1] for tra_iii in tra_ii])

        vid_scalar_tracks_all_array[ii,tra_ii_times] = tra_ii_vals.copy()

    return vid_scalar_tracks_all_array



def unpad_bbox_tracks(all_uniq_tracks, pad, shape, nframes, aspect_ratio=3): 
    
    import numpy as np 
    # define the bounds. 
    bounds_x = [0, shape[1]-1]
    bounds_y = [0, shape[0]-1] 
    
    all_uniq_bbox_tracks_new = []
    
    for tra_ii in np.arange(len(all_uniq_tracks)):

        all_uniq_tracks_ii = all_uniq_tracks[tra_ii]
        # all_track_labels_ii = track_labels[tra_ii]
        all_uniq_bbox_track_ii = []

        for jj in np.arange(len(all_uniq_tracks_ii)):
            
            tra = all_uniq_tracks_ii[jj]
            # tra_label = all_track_labels_ii[jj][-1]
            tp, conf, bbox = tra
            x1, y1, x2, y2 = bbox
            
            x1 = np.clip(x1, 0, shape[1]-1+2*pad)
            x2 = np.clip(x2, 0, shape[1]-1+2*pad)
            y1 = np.clip(y1, 0, shape[0]-1+2*pad)
            y2 = np.clip(y2, 0, shape[0]-1+2*pad)
    
            x1_= x1-pad 
            y1_= y1-pad 
            x2_ = x2-pad
            y2_ = y2-pad
            # print(x1_, x2_, y1_, y2_)
            
            """
            This is more complex ... 
                if y < 0 or  y > shape[0], then we have to flip x coordinate and flip y-coordinate.  
                # test this inversion ! ######## ---> for a fixed timepoint, just to double ceck we are getting the correct alogrithm. !. 
            """
            # left top 
            if y1_ < bounds_y[0] and (x1_ < bounds_x[0]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y1_ = -1 * y1_  # flip y 
                # x1_ = (bounds_x[1]+1*pad + x1_) - bounds_x[1] # flip x axis. 
                x1_ = bounds_x[0] - x1_ 
    
            if y2_ < bounds_y[0] and (x2_ < bounds_x[0]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y2_ = -1 * y2_ 
                # x2_ = (bounds_x[1]+1*pad + x2_) - bounds_x[1] # flip x axis. 
                x2_ = bounds_x[0] - x2_ 
            # middle top # this is ok 
            if y1_ < bounds_y[0] and (x1_ >= bounds_x[0] and x1_ <= bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y1_ = -1 * y1_ 
                x1_ = bounds_x[1] - x1_ # flip x axis. 
    
            if y2_ < bounds_y[0] and (x2_ >= bounds_x[0] and x2_ <= bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y2_ = -1 * y2_ 
                x2_ = bounds_x[1] - x2_ # flip x axis. 
                
            # right top 
            if y1_ < bounds_y[0] and (x1_ > bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y1_ = -1 * y1_  # flip y 
                x1_ = bounds_x[1] - (x1_ - bounds_x[1])
    
            if y2_ < bounds_y[0] and (x2_ > bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y2_ = -1 * y2_ 
                x2_ = bounds_x[1] - (x2_ - bounds_x[1])
    
            # middle left 
            if (y1_ >= bounds_y[0] and y1_ <= bounds_y[1]) and (x1_ < bounds_x[0]):
                # simply shift the x1_ back 
                x1_ = x1_ + (bounds_x[1]+1)
            if (y2_ >= bounds_y[0] and y2_ <= bounds_y[1]) and (x2_ < bounds_x[0]): 
                x2_ = x2_ + (bounds_x[1]+1)
    
            # middle right
            if (y1_ >= bounds_y[0] and y1_ <= bounds_y[1]) and (x1_ > bounds_x[1]):
                # simply shift the x1_ back 
                x1_ = x1_ - (bounds_x[1]+1)
            if (y2_ >= bounds_y[0] and y2_ <= bounds_y[1]) and (x2_ > bounds_x[1]):
                # simply shift the x1_ back 
                x2_ = x2_ - (bounds_x[1]+1)
    
            # left bottom 
            if y1_ > bounds_y[1] and (x1_ < bounds_x[0]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y1_ = bounds_y[1] + (bounds_y[1] - y1_) # flipping at the other end.  
                # x1_ = (bounds_x[1]+1*p-d + x1_) - bounds_x[1]  # flip x axis.
                x1_ = bounds_x[0] - x1_ 
            if y2_ > bounds_y[1] and (x2_ < bounds_x[0]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y2_ = bounds_y[1] + (bounds_y[1] - y2_) # flipping at the other end. 
                x2_ = bounds_x[0] - x2_ 
                # x2_ = (bounds_x[1]+1*pad + x2_) - bounds_x[1]
                # x2_ = x2_ - (bounds_x[1]+1)
                
            # middle bottom 
            if y1_ > bounds_y[1] and (x1_ >= bounds_x[0] and x1_ <= bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y1_ = bounds_y[1] + (bounds_y[1] - y1_) # flipping at the other end.  
                x1_ = bounds_x[1] - x1_ # flip x axis. 
    
            if y2_ > bounds_y[1] and (x2_ >= bounds_x[0] and x2_ <= bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y2_ = bounds_y[1] + (bounds_y[1] - y2_) # flipping at the other end. 
                x2_ = bounds_x[1] - x2_ # flip x axis. 
    
            # right bottom  
            if y1_ > bounds_y[1] and (x1_ > bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y1_ = bounds_y[1] + (bounds_y[1] - y1_) # flipping at the other end.  
                # x1_ = x1_ - (bounds_x[1]+1*pad) + bounds_x[1] # flip x axis. 
                x1_ = bounds_x[1] - (x1_ - bounds_x[1])
            if y2_ > bounds_y[1] and (x2_ > bounds_x[1]):
                # then we flip the x axis and flip the y coordinate into the shape. (=180 rotation) 
                y2_ = bounds_y[1] + (bounds_y[1] - y2_) # flipping at the other end. 
                # x2_ = x2_ - (bounds_x[1]+1*pad) + bounds_x[1] # flip x axis. 
                x2_ = bounds_x[1] - (x2_ - bounds_x[1])
    
            """
            need a accuracy test for the designated label if the label is not the background label!. 
            """
            # we check the sign change here! to know if we need to modify.  
            w1_ = (x2_ - x1_) # x2 must be higher? if negative that means x1 needs to be shifted back. 
            h1_ = (y2_ - y1_) # if negative means y1 needs to be shifted 
            # print(w1_, h1_, (x1_,y1_,x2_,y2_))
            if w1_ < 0:
                # we have to shift but dunno which way. 
                x1_test = np.clip(x1_ - (bounds_x[1]), 0, bounds_x[1])
                x2_test = np.clip(x2_ + (bounds_x[1]), 0, bounds_x[1])
    
                area_x1_test = np.abs(x2_ - x1_test) * np.abs(y2_-y1_)
                area_x2_test = np.abs(x2_test - x1_) * np.abs(y2_-y1_)
                
                if area_x1_test > area_x2_test:
                    x1_ = x1_test
                else:
                    x2_ = x2_test
    
            if h1_ < 0:
                # y1_ = np.clip(y1_ - (bounds_y[1]+1), 0, bounds_y[1])
                # y2_ = np.clip(y2_ + (bounds_y[1]+1), 0, bounds_y[1])
                # we have to shift but dunno which way. 
                y1_test = np.clip(y1_ - (bounds_y[1]), 0, bounds_y[1]) # make y1 smaller or 
                y2_test = np.clip(y2_ + (bounds_y[1]), 0, bounds_y[1]) # make y2 bigger
    
                area_y1_test = np.abs(y2_ - y1_test) * np.abs(x2_ - x1_)
                area_y2_test = np.abs(y2_test - y1_) * np.abs(x2_ - x1_)
                
                if area_y1_test > area_y2_test:
                    y1_ = y1_test
                else:
                    y2_ = y2_test
            
            bbox_new = np.hstack([x1_, y1_, x2_, y2_])
            all_uniq_bbox_track_ii.append([tp, conf, bbox_new])
        all_uniq_bbox_tracks_new.append(all_uniq_bbox_track_ii)

        # convert this !. 
        # compile into a numpy array and send this out too 
        _, all_uniq_bbox_tracks_array_new = bbox_tracks_2_array(all_uniq_bbox_tracks_new, nframes=nframes, ndim=4)
        all_uniq_bbox_tracks_centroids_xy_new = np.array([.5*(all_uniq_bbox_tracks_array_new[...,0] + all_uniq_bbox_tracks_array_new[...,2]), 
                                                          .5*(all_uniq_bbox_tracks_array_new[...,1] + all_uniq_bbox_tracks_array_new[...,3])])
        
    return all_uniq_bbox_tracks_new, all_uniq_bbox_tracks_array_new, all_uniq_bbox_tracks_centroids_xy_new


def assign_label_detection_to_track_frame_by_frame(bbox_array_time, bbox_detections_time, min_iou=.25):

    """
    bbox_array_time: N_tracks x N_frames x 4
    bbox_detections_time: N_frames list 
    """
    import numpy as np 
    from scipy.optimize import linear_sum_assignment

    n_tracks, n_frames, _ = bbox_array_time.shape
    bbox_array_time_labels = np.zeros((n_tracks, n_frames), dtype=np.float64) # initialise to background!. 
    bbox_array_time_labels[:] = np.nan # initialise to empty 

    # iterate over time 
    for tt in np.arange(n_frames)[:]:
        # reference
        bbox_frame_tt = bbox_detections_time[tt].copy()
        bbox_labels_tt = bbox_frame_tt[:,0].copy()
        bbox_labels_bbox = bbox_frame_tt[:,1:5].copy()

        tracks_bboxes = bbox_array_time[:,tt].copy()
        non_nan_select = np.logical_not(np.isnan(tracks_bboxes[:,0])) # just test the one coordinate. # only these need to be given a label!. 

        tracks_bboxes_bbox = tracks_bboxes[non_nan_select>0].copy()
        labels_non_nan_select = np.zeros(len(tracks_bboxes_bbox), dtype=np.int) # preinitialise 

        # build the iou cost matrix. between rows: tracks and cols: the frame boxes. 
        iou_matrix = bbox_iou_corner_xy(tracks_bboxes_bbox, 
                                        bbox_labels_bbox) 
        iou_matrix = np.clip(1.-iou_matrix, 0, 1) # to make it a dissimilarity matrix. 
        # solve the pairing problem.
        ind_ii, ind_jj = linear_sum_assignment(iou_matrix)
        
        # threshold as the matching is maximal. 
        iou_ii_jj = iou_matrix[ind_ii, ind_jj].copy()
        keep = iou_ii_jj <= (1-min_iou) # keep those with a minimal distance. 

        # keep = iou_ii_jj <= dist_thresh
        ind_ii = ind_ii[keep>0]; 
        ind_jj = ind_jj[keep>0]
        
        labels_non_nan_select[ind_ii] = bbox_labels_tt[ind_jj].copy()

        # copy back into the labels. 
        bbox_array_time_labels[non_nan_select>0, tt] = labels_non_nan_select.copy() # copy these assignments in now. 

    return bbox_array_time_labels


def compute_labeled_to_unlabeled_track(track_label_array):

    import numpy as np 

    n_tracks, n_frames = track_label_array.shape

    track_counts = []
    for tra_ii in np.arange(n_tracks):
        tra_label = track_label_array[tra_ii].copy()
        valid = np.logical_not(np.isnan(tra_label))

        num_valid = np.sum(valid)
        valid_labels = tra_label[valid].astype(np.int)

        num_nonzeros = np.sum(valid_labels > 0) 
        track_counts.append([num_valid, num_nonzeros, float(num_nonzeros) / num_valid, n_frames])
    track_counts = np.vstack(track_counts)

    return track_counts


def moving_average_bbox_tracks(list_tracks, avg_func=np.nanmean, winsize=3, min_winsize_prop=0.1, pad_mode='edge', *args):

    import numpy as np 

    list_tracks_smooth = []

    for tra_ii in np.arange(len(list_tracks)):

        track_ii = np.array(list_tracks[tra_ii]) # numpy array 
        winsize_track = winsize
        if winsize/float(len(track_ii)) < min_winsize_prop: 
            # then reduce the winsize smoothing!.   
            winsize_track = np.maximum(3, int(len(track_ii)*min_winsize_prop))
        track_ii_pad = np.pad(track_ii, [[winsize_track//2, winsize_track//2], [0,0]], mode=pad_mode, *args)

        track_ii_smooth = np.vstack([avg_func(track_ii_pad[tt:tt+winsize_track], axis=0) for tt in np.arange(len(track_ii))])
        list_tracks_smooth.append(track_ii_smooth)

    return list_tracks_smooth


# using optical flow to implement a predictive tracker for blebs. 
def track_bleb_bbox(vid_flow, # flow is already computed. 
                    vid_bboxes, # bbox files for each frame only !.   
                    vid=None, # only for visualisation purposes. 
                    iou_match=.25,
                    ds_factor = 1,
                    wait_time=10,
                    min_aspect_ratio=3,
                    max_dense_bbox_cover=8,
                    min_tra_len_filter=5,
                    min_tra_lifetime_ratio = .1, 
                    to_viz=True,
                    remove_eccentric_bbox=False,
                    saveanalysisfolder=None):

    """
    uses optical flow and registration as a predictor to improve tracking over occlusion.
    # we should preserve the bbox coordinates + the given probability from yolo.....  that way we can always do further processing? 
    """
    import numpy as np 
    import seaborn as sns
    import pylab as plt 
    import os 
    from tqdm import tqdm 
    from scipy.optimize import linear_sum_assignment

    """
    initial setting up. 
    """
    im_shape = vid_flow[0].shape[:2]
    nframes = len(vid_flow)+1 # the number of frames in the video.

    if saveanalysisfolder is not None:
        print('saving graphics in folder: ', saveanalysisfolder)
        # saveanalysisfolder_movie = os.path.join(saveanalysisfolder, 'track_boundaries'); mkdir(saveanalysisfolder_movie); 
        saveanalysisfolder_movie_bbox = os.path.join(saveanalysisfolder, 'track_bbox'); 
        mkdir(saveanalysisfolder_movie_bbox);     

    # =============================================================================
    #     BBox Tracks
    # =============================================================================

    # terminated_vid_cell_tracks = [] 
    terminated_vid_bbox_tracks = []
    terminated_check_match_bbox_tracks = []

    vid_bbox_tracks = [] # this is to keep track of the actual bbox we use, including inferred. 
    vid_bbox_tracks_last_time = [] # record when the last time a proper match was made. 
    vid_match_check_bbox_tracks = []

# =============================================================================
#     build up the tracks dynamically now, frame by frame dynamically , with a track waiting time before termination. 
# ============================================================================
    for ii in tqdm(np.arange(nframes-1)[:]):
        """
        if working set is empty or otherwise, then add to working set. 
        """
        # add these to the working tracks. 
        if ii == 0 or len(vid_bbox_tracks)==0: 
            # bboxfile_ii = vid_bboxes[ii]
            """
            swap this element out. 
            """
            # prob_ii, boxes_ii = load_bbox_frame_voc( vid[0], bboxfile_ii) # modify this here!. 
            boxes_ii = vid_bboxes[ii].copy()
            # set prob dependent on the size of the boxes. (if 5D then take the first as a measure of objectness else assume 1.)
            if boxes_ii.shape[-1] == 4: 
                prob_ii = np.ones(len(boxes_ii)) 
            else:
                prob_ii = boxes_ii[:,0].copy()
                boxes_ii = boxes_ii[:,1:].copy()

            boxes_ii = boxes_ii / float(ds_factor)
            
            # clip
            boxes_ii[:,0] = np.clip(boxes_ii[:,0], 0, im_shape[1]-1)
            boxes_ii[:,1] = np.clip(boxes_ii[:,1], 0, im_shape[0]-1)
            boxes_ii[:,2] = np.clip(boxes_ii[:,2], 0, im_shape[1]-1)
            boxes_ii[:,3] = np.clip(boxes_ii[:,3], 0, im_shape[0]-1)
        
            # remove all boxes that are have an area 1 pixels or less.               
            boxes_ii_w = boxes_ii[:,2] - boxes_ii[:,0]
            boxes_ii_h = boxes_ii[:,3] - boxes_ii[:,1]
            boxes_ii = boxes_ii[boxes_ii_w*boxes_ii_h>0]

            # suppress implausible. 
            if remove_eccentric_bbox:
              boxes_ii, keep_ind_ii = remove_very_large_bbox(boxes_ii, 
                                                            im_shape, 
                                                            thresh=0.5, 
                                                            aspect_ratio = 3, # we don't expect it to be very long. 
                                                            method='density', 
                                                            max_density=max_dense_bbox_cover, 
                                                            mode='fast', 
                                                            return_ind=True)
            else:
              keep_ind_ii = np.ones(len(boxes_ii))>0

            assert(len(boxes_ii) == len(keep_ind_ii)) # keep a check here.       
            prob_ii = prob_ii[keep_ind_ii]

            for jj in np.arange(len(boxes_ii)):
                vid_bbox_tracks.append([[ii, prob_ii[jj], boxes_ii[jj]]]) # add the prob in as another entry here. 
                vid_bbox_tracks_last_time.append(ii) # update with current time. 
                vid_match_check_bbox_tracks.append([[ii, 1]]) # update with the current time., this is just to select whether this box was propagated or found in the original detections. 
                
        """
        load the current working set of tracks. 
        """
        # 1) check for track termination 
        # get the last timepoint. 
        boxes_ii_track_time = np.hstack(vid_bbox_tracks_last_time)
        
        # check if any of the box tracks need to be terminated. 
        track_terminate =  boxes_ii_track_time < ii - wait_time
        track_terminate_id = np.arange(len(track_terminate))[track_terminate]

        if len(track_terminate_id)>0:
            # update the relevant info.
            for idd in track_terminate_id:
                #update
                terminated_vid_bbox_tracks.append(vid_bbox_tracks[idd][:-wait_time-1])
                terminated_check_match_bbox_tracks.append(vid_match_check_bbox_tracks[idd][:-wait_time-1])
                # terminated_vid_cell_tracks_properties.append(vid_cell_tracks_properties[idd][:-wait_time-1])

            # # reform the working set. 
            # vid_cell_tracks = [vid_cell_tracks[jjj] for jjj in np.arange(len(vid_cell_tracks)) if jjj not in track_terminate_id] # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks = [vid_bbox_tracks[jjj] for jjj in np.arange(len(vid_bbox_tracks)) if jjj not in track_terminate_id] # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties = [vid_cell_tracks_properties[jjj] for jjj in np.arange(len(vid_cell_tracks_properties)) if jjj not in track_terminate_id]
            vid_bbox_tracks_last_time = [vid_bbox_tracks_last_time[jjj] for jjj in np.arange(len(vid_bbox_tracks_last_time)) if jjj not in track_terminate_id]
            vid_match_check_bbox_tracks = [vid_match_check_bbox_tracks[jjj] for jjj in np.arange(len(vid_match_check_bbox_tracks)) if jjj not in track_terminate_id]

        # load the current version of the boxes. 
        boxes_ii_track = np.array([bb[-1][-1] for bb in vid_bbox_tracks]) # the bboxes to consider.
        boxes_ii_track_prob = np.array([bb[-1][1] for bb in vid_bbox_tracks]) 
        boxes_ii_track_time = np.array([bb[-1][0] for bb in vid_bbox_tracks]) # the time of the last track. 

        """
        Infer the next frame boxes from current boxes using optical flow.    
        """
        boxes_ii_track_pred = []
        
        for jjj in np.arange(len(boxes_ii_track)):
            """
            Predict using the flow the likely position of boxes. 
            """
            new_tfs, boxes_ii_pred = predict_new_boxes_flow_tf(boxes_ii_track[jjj][None,:], 
                                                               vid_flow[boxes_ii_track_time[jjj]])
            # clip  
            boxes_ii_pred[:,0] = np.clip(boxes_ii_pred[:,0], 0, im_shape[1]-1)
            boxes_ii_pred[:,1] = np.clip(boxes_ii_pred[:,1], 0, im_shape[0]-1)
            boxes_ii_pred[:,2] = np.clip(boxes_ii_pred[:,2], 0, im_shape[1]-1)
            boxes_ii_pred[:,3] = np.clip(boxes_ii_pred[:,3], 0, im_shape[0]-1)
            boxes_ii_track_pred.append(boxes_ii_pred[0])
  
        boxes_ii_track_pred = np.array(boxes_ii_track_pred)
            
        """
        load the next frame boxes. which are the candidates.  
        """
        # bboxfile_jj = vid_bboxes[ii+1]
        # prob_jj, boxes_jj = load_bbox_frame_voc( vid[ii+1], 
                                                 # bboxfile_jj)
        boxes_jj = vid_bboxes[ii+1].copy();
        # set prob dependent on the size of the boxes. (if 5D then take the first as a measure of objectness else assume 1.)
        if boxes_jj.shape[-1] == 4: 
            prob_jj = np.ones(len(boxes_jj))
        else:
            prob_jj = boxes_jj[:,0].copy()
            boxes_jj = boxes_jj[:,1:].copy()
        
        boxes_jj = boxes_jj/float(ds_factor)

        # clip
        boxes_jj[:,0] = np.clip(boxes_jj[:,0], 0, im_shape[1]-1)
        boxes_jj[:,1] = np.clip(boxes_jj[:,1], 0, im_shape[0]-1)
        boxes_jj[:,2] = np.clip(boxes_jj[:,2], 0, im_shape[1]-1)
        boxes_jj[:,3] = np.clip(boxes_jj[:,3], 0, im_shape[0]-1) 
        # remove all boxes that are have an area 1 pixels or less.               
        boxes_jj_w = boxes_jj[:,2] - boxes_jj[:,0]
        boxes_jj_h = boxes_jj[:,3] - boxes_jj[:,1]
        boxes_jj = boxes_jj[boxes_jj_w*boxes_jj_h>0]

        # suppress implausible. 
        if remove_eccentric_bbox:
          boxes_jj, keep_ind_jj = remove_very_large_bbox(boxes_jj, 
                                                      im_shape, 
                                                      thresh=0.5, 
                                                      aspect_ratio = 3, # we don't expect it to be very long. 
                                                      method='density', 
                                                      max_density=max_dense_bbox_cover, 
                                                      mode='fast', return_ind=True)
        else:
          keep_ind_jj = np.ones(len(boxes_jj))>0
              
        prob_jj = prob_jj[keep_ind_jj]
        
        """
        build the association matrix and match boxes based on iou. 
        """
        iou_matrix = bbox_iou_corner_xy(boxes_ii_track_pred, 
                                        boxes_jj); 
        iou_matrix = np.clip(1.-iou_matrix, 0, 1) # to make it a dissimilarity matrix. 
        
        # solve the pairing problem.
        ind_ii, ind_jj = linear_sum_assignment(iou_matrix)
        
        # threshold as the matching is maximal. 
        iou_ii_jj = iou_matrix[ind_ii, ind_jj].copy()
        keep = iou_ii_jj <= (1-iou_match)
        # keep = iou_ii_jj <= dist_thresh
        ind_ii = ind_ii[keep>0]; 
        ind_jj = ind_jj[keep>0]
        
        """
        Update the trajectories. 
        """
        # update first the matched.
        for match_ii in np.arange(len(ind_ii)):
            # vid_cell_tracks[ind_ii[match_ii]].append([ii+1, ind_jj[match_ii]]) # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks[ind_ii[match_ii]].append([ii+1, prob_jj[ind_jj[match_ii]], boxes_jj[ind_jj[match_ii]]]) # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties[ind_ii[match_ii]].append([ii+1, masks_jj_properties[ind_jj[match_ii]]]) # append the properties of the next time point. 
            vid_bbox_tracks_last_time[ind_ii[match_ii]] = ii+1 # let this be a record of the last time a 'real' segmentation was matched, not one inferred from optical flow. 
            # vid_mask_tracks_last[ind_ii[match_ii]] = masks_jj[...,ind_jj[match_ii]] # retain just the last masks thats relevant for us. 
            # vid_mask_tracks_last[ind_ii[match_ii]] = boxes_jj[ind_jj[match_ii]]
            vid_match_check_bbox_tracks[ind_ii[match_ii]].append([ii+1, 1]) # success, append 0 
            
        no_match_ind_ii = np.setdiff1d(np.arange(len(boxes_ii_track_pred)), ind_ii)
        no_match_ind_jj = np.setdiff1d(np.arange(len(boxes_jj)), ind_jj)
        
        for idd in no_match_ind_ii:
            # these tracks already exist so we just add to the existant tracks.         
            # vid_cell_tracks[idd].append([ii+1, -1]) # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks[idd].append([ii+1, boxes_ii_track_prob[idd], boxes_ii_track_pred[idd]]) # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties[idd].append([ii+1, properties_ii_track_pred[idd]])
            # vid_bbox_tracks_last_time[idd] = ii+1 # let this be a record of the last time a 'real' segmentation was matched, not one inferred from optical flow. 
            # vid_mask_tracks_last[idd] = boxes_ii_track_pred[idd] # retain just the last masks thats relevant for us.
            vid_match_check_bbox_tracks[idd].append([ii+1, 0]) # no success, append 0  
    
        for idd in no_match_ind_jj:
            # these tracks don't exsit yet so we need to create new tracks. 
            # vid_cell_tracks.append([[ii+1, idd]]) # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks.append([[ii+1, prob_jj[idd], boxes_jj[idd]]]) # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties.append([[ii+1, masks_jj_properties[idd]]])
            vid_bbox_tracks_last_time.append(ii+1) # let this be a record of the last time a 'real' segmentation was matched, not one inferred from optical flow. 
            # vid_mask_tracks_last.append(masks_jj[...,idd]) # retain just the last masks thats relevant for us. 
            # vid_mask_tracks_last.append(boxes_jj[idd])
            vid_match_check_bbox_tracks.append([[ii+1, 1]])
                
# =============================================================================
#     Combine the tracks togther
# =============================================================================
    vid_bbox_tracks_all = terminated_vid_bbox_tracks + vid_bbox_tracks # combine
    vid_match_checks_all = terminated_check_match_bbox_tracks + vid_match_check_bbox_tracks


# =============================================================================
#     Compute some basic track properties for later filtering. 
# =============================================================================    
    vid_bbox_tracks_all_lens = np.hstack([len(tra) for tra in vid_bbox_tracks_all])
    vid_bbox_tracks_all_start_time = np.hstack([tra[0][0] for tra in vid_bbox_tracks_all])
    
    print(vid_bbox_tracks_all_lens)
    print(vid_bbox_tracks_all_start_time)
    
    vid_bbox_tracks_lifetime_ratios = vid_bbox_tracks_all_lens / (len(vid_flow) - vid_bbox_tracks_all_start_time).astype(np.float)
    
# =============================================================================
#     Turn into a proper array of n_tracks x n_time x 4 or 5.... 
# =============================================================================
    vid_bbox_tracks_prob_all_array, vid_bbox_tracks_all_array = bbox_tracks_2_array(vid_bbox_tracks_all, nframes=nframes, ndim=boxes_ii.shape[1])
    vid_match_checks_all_array = bbox_scalar_2_array(vid_match_checks_all, nframes=nframes)

# =============================================================================
#   Apply the given filtering parameters for visualization else it will be messy.        
# =============================================================================

    if to_viz:

        fig, ax = plt.subplots()
        ax.scatter(vid_bbox_tracks_all_start_time, 
                   vid_bbox_tracks_all_lens, 
                   c=vid_bbox_tracks_lifetime_ratios, 
                   vmin=0, 
                   vmax=1, cmap='coolwarm')
        plt.show()


        # keep filter
        keep_filter = np.logical_and(vid_bbox_tracks_all_lens>=min_tra_len_filter, 
                                     vid_bbox_tracks_lifetime_ratios>=min_tra_lifetime_ratio)
        keep_ids = np.arange(len(vid_bbox_tracks_all_lens))[keep_filter]

        plot_vid_bbox_tracks_all = vid_bbox_tracks_all_array[keep_ids].copy()
        plot_colors = sns.color_palette('hls', len(plot_vid_bbox_tracks_all))

        # print(len(plot_vid_bbox_tracks_all), len(vid_bbox_tracks_all_array))
        # iterate over time. 
        for frame_no in np.arange(nframes):

            fig, ax = plt.subplots(figsize=(5,5))
            plt.title('Frame: %d' %(frame_no+1))
            
            if vid is not None:
                vid_overlay = vid[frame_no].copy()
            else:
                vid_overlay = np.zeros(im_shape)

            ax.imshow(vid_overlay, cmap='gray')
            
            for ii in np.arange(len(plot_vid_bbox_tracks_all))[:]:
                bbox_tra_ii = plot_vid_bbox_tracks_all[ii][frame_no] # fetch it at this point in time. 

                if ~np.isnan(bbox_tra_ii[0]):
                    # then there is a valid bounding box. 
                    x1,y1,x2,y2 = bbox_tra_ii 
                    ax.plot( [x1,x2,x2,x1,x1], 
                             [y1,y1,y2,y2,y1], lw=1, color = plot_colors[ii])

            ax.set_xlim([0, im_shape[1]-1])
            ax.set_ylim([im_shape[0]-1, 0])
            plt.axis('off')
            plt.grid('off')
            
            if saveanalysisfolder is not None:
                fig.savefig(os.path.join(saveanalysisfolder_movie_bbox, 'Frame-%s.png' %(str(frame_no).zfill(3))), bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
    # spio.savemat(savematfile, 
    #                 {'boundaries':organoid_boundaries, 
    #                 'initial_seg': seg0_grab,
    #                 'motion_source_map': spixels_B_motion_sources,
    #                 'seg_pts': organoid_segs_pts})

    # return organoid_boundaries, organoid_segs_pts
    return vid_bbox_tracks_prob_all_array, vid_bbox_tracks_all, vid_bbox_tracks_all_array, vid_match_checks_all_array, (vid_bbox_tracks_all_lens, vid_bbox_tracks_all_start_time, vid_bbox_tracks_lifetime_ratios)


def tracks2Darray_to_3Darray(tracks2D_time, uv_params_time):

    from ..Unzipping import unzip_new as uzip
    import numpy as np 

    m, n = uv_params_time.shape[1:-1]
    n_tracks, n_time, _ = tracks2D_time.shape
    tracks3D_time = np.zeros((n_tracks, n_time, 3))
    tracks3D_time[:] = np.nan 

    for tt in np.arange(n_time):
        # iterate over frame
        uv_params = uv_params_time[tt].copy()
        tracks2D_tt = tracks2D_time[:,tt].copy()

        # only interpolate for non nan vals. 
        val_track = np.logical_not(np.isnan(tracks2D_tt[:,0]))

        pts2D = tracks2D_tt[val_track>0].copy()
        pts2D[...,0] = np.clip(pts2D[...,0], 0, n-1) # this assumes x,y... 
        pts2D[...,1] = np.clip(pts2D[...,1], 0, m-1)

        # is this interpolation bad? 
        pts3D = np.array([uzip.map_intensity_interp2(pts2D[:,::-1], 
                                                     grid_shape=(m,n), 
                                                     I_ref=uv_params[...,ch], 
                                                     method='linear', 
                                                     cast_uint8=False, s=0) for ch in np.arange(uv_params.shape[-1])]).T
        tracks3D_time[val_track>0, tt, :] = pts3D.copy()

    return tracks3D_time



# =============================================================================
#   Add some track postprocessing support namely removal of all nan tracks and non-max suppression amongst tracks. (operating purely on bboxes!)
# =============================================================================

def calculate_iou_matrix_time(box_arr1, box_arr2, eps=1e-9):
        
    import numpy as np 
    x11 = box_arr1[...,0]; y11 = box_arr1[...,1]; x12 = box_arr1[...,2]; y12 = box_arr1[...,3]
    x21 = box_arr2[...,0]; y21 = box_arr2[...,1]; x22 = box_arr2[...,2]; y22 = box_arr2[...,3]
    m,n = x11.shape
    # # n_tracks x n_time. 
    # flip this.
    x11 = x11.T; y11 = y11.T; x12 = x12.T; y12 = y12.T
    x21 = x21.T; y21 = y21.T; x22 = x22.T; y22 = y22.T 

    xA = np.maximum(x11[...,None], x21[:,None,:])
    yA = np.maximum(y11[...,None], y21[:,None,:])
    xB = np.minimum(x12[...,None], x22[:,None,:])
    yB = np.minimum(y12[...,None], y22[:,None,:])
    
    interArea = np.maximum((xB - xA + eps), 0) * np.maximum((yB - yA + eps), 0)
    boxAArea = (x12 - x11 + eps) * (y12 - y11 + eps)
    boxBArea = (x22 - x21 + eps) * (y22 - y21 + eps)

    # iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    iou = interArea / (boxAArea[...,None] + boxBArea[:,None,:] - interArea)
    # # iou = iou.reshape((m,n))
    return iou

# replace with iou bbox_tracks
def iou_boundary_tracks(tra_1, tra_2):

    import numpy as np 

    n_pts = len(tra_1) 
    iou_time = np.zeros(n_pts)

    for ii in range(n_pts):
        tra_1_ii = tra_1[ii]
        tra_2_ii = tra_2[ii]

        if np.isnan(tra_1_ii[0,0]) or np.isnan(tra_2_ii[0,0]):
            iou_time[ii] = np.nan
        else:
            x1, x2 = np.min(tra_1_ii[...,1]), np.max(tra_1_ii[...,1])
            y1, y2 = np.min(tra_1_ii[...,0]), np.max(tra_1_ii[...,0])
            
            x1_, x2_ = np.min(tra_2_ii[...,1]), np.max(tra_2_ii[...,1])
            y1_, y2_ = np.min(tra_2_ii[...,0]), np.max(tra_2_ii[...,0])
            
            bbox1 = np.hstack([x1,y1,x2,y2])
            bbox2 = np.hstack([x1_,y1_,x2_,y2_])
            
            print(bbox1, bbox2)
            iou_12 = get_iou(bbox1, bbox2)
            iou_time[ii] = iou_12

    return iou_time

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    if np.sum(bb1) < 1 and np.sum(bb2) < 1:
        iou = 0
    else:
        bb1 = {'x1': bb1[0], 
               'y1': bb1[1], 
               'x2': bb1[2],
               'y2': bb1[3]}
        bb2 = {'x1': bb2[0], 
               'y1': bb2[1], 
               'x2': bb2[2],
               'y2': bb2[3]}
        
    #    print(bb1)
    #    print(bb2)
    #     test for 0
            
        # allow for collapsed boxes.           
        assert bb1['x1'] <= bb1['x2']
        assert bb1['y1'] <= bb1['y2']
        assert bb2['x1'] <= bb2['x2']
        assert bb2['y1'] <= bb2['y2']
    
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
    
        if x_right < x_left or y_bottom < y_top:
            return 0.0
    
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
    return iou

def iou_bbox_tracks(tra_1, tra_2):

    import numpy as np 

    n_pts = len(tra_1) 
    iou_time = np.zeros(n_pts)

    for ii in range(n_pts):
        tra_1_ii = tra_1[ii] # get the track at a particular timepoint should be 4
        tra_2_ii = tra_2[ii]

        if np.isnan(tra_1_ii[0,0]) or np.isnan(tra_2_ii[0,0]):
            iou_time[ii] = np.nan # prop the nan onwards. 
        else:
            x1, y1, x2, y2 = tra_1_ii
            x1_, y1_, x2_, y2_ = tra_2_ii
            # x1, x2 = np.min(tra_1_ii[...,1]), np.max(tra_1_ii[...,1])
            # y1, y2 = np.min(tra_1_ii[...,0]), np.max(tra_1_ii[...,0])
            
            # x1_, x2_ = np.min(tra_2_ii[...,1]), np.max(tra_2_ii[...,1])
            # y1_, y2_ = np.min(tra_2_ii[...,0]), np.max(tra_2_ii[...,0])
            bbox1 = np.hstack([x1,y1,x2,y2])
            bbox2 = np.hstack([x1_,y1_,x2_,y2_])
            
            # print(bbox1, bbox2)
            iou_12 = get_iou(bbox1, bbox2)
            iou_time[ii] = iou_12

    return iou_time


def pairwise_iou_tracks(boundaries_list):
    
    import itertools
    import numpy as np 
    
    Ns = np.hstack([len(bb) for bb in boundaries_list]) # this is for dissecting IDs.
    
    ind_ids = np.hstack([[str(jj)+'_'+str(ii) for jj in range(Ns[ii])] for ii in range(len(Ns))])
    # stack all of them together. 
    all_boundaries = np.vstack(boundaries_list)
    
    # print(all_boundaries.shape)
    sim_matrix = np.zeros((len(all_boundaries), len(all_boundaries)))
    shared_time_matrix = np.zeros((len(all_boundaries), len(all_boundaries)))
    
    for i, j in itertools.combinations(range(len(all_boundaries)),2):
        
        boundary_i = all_boundaries[i]
        boundary_j = all_boundaries[j]
        
        # iou_track = iou_boundary_tracks(boundary_i, boundary_j)
        iou_track = iou_bbox_tracks(boundary_i, boundary_j)
        
        sim_matrix[i,j] = np.nanmean(iou_track)
        sim_matrix[j,i] = np.nanmean(iou_track)
        
        shared_time_matrix[i,j] = ~np.isnan(iou_track).sum()
        shared_time_matrix[j,i] = ~np.isnan(iou_track).sum()
        
    return ind_ids, (sim_matrix, shared_time_matrix)


def pairwise_iou_tracks_fast(boundaries_list, eps=1e-9, return_bbox=False, avg_func=np.nanmean):
        
    import numpy as np 
    import time 
    
    Ns = np.hstack([len(bb) for bb in boundaries_list]) # this is for dissecting IDs.
    
    ind_ids = np.hstack([[str(jj)+'_'+str(ii) for jj in range(Ns[ii])] for ii in range(len(Ns))])
    # stack all of them together. 
    all_boundaries = np.vstack(boundaries_list) # flatten all. 
    
    # # turn the boundaries into bbox. 
    # all_boundaries_bbox_xmin = np.min(all_boundaries[...,0], axis=-1)
    # all_boundaries_bbox_xmax = np.max(all_boundaries[...,2], axis=-1)
    # all_boundaries_bbox_ymin = np.min(all_boundaries[...,1], axis=-1)
    # all_boundaries_bbox_ymax = np.max(all_boundaries[...,3], axis=-1)
    
    # all_boundaries_bbox = np.concatenate([all_boundaries_bbox_xmin[...,None], 
    #                                       all_boundaries_bbox_ymin[...,None], 
    #                                       all_boundaries_bbox_xmax[...,None], 
    #                                       all_boundaries_bbox_ymax[...,None]], axis=-1)
    all_boundaries_bbox = all_boundaries.copy()
    
    all_sim_matrix = calculate_iou_matrix_time(all_boundaries_bbox, 
                                                all_boundaries_bbox, 
                                            eps=eps)
    
    sim_matrix = avg_func(all_sim_matrix, axis=0)
    shared_time_matrix = np.sum(~np.isnan(all_sim_matrix), axis=0)
     
    if return_bbox:
        return ind_ids, (sim_matrix, shared_time_matrix), (all_boundaries_bbox, sim_matrix)
    else:
        return ind_ids, (sim_matrix, shared_time_matrix)

def unique_pts(a):
    import numpy as np 
    # returns the unique rows of an array. 
    return np.vstack(list({tuple(row) for row in a}))

def get_id_cliques(id_list):
    """
    given a list of identities, merges the common ids into a cluster/clique
    """
    N = len(id_list)
    cliques = [id_list[0]]
    
    for ii in range(1, N):
        
        id_list_ii = id_list[ii]
        
        add_clique = True
        for cc_i, cc in enumerate(cliques):
            if len(np.intersect1d(id_list_ii, cc)) > 0:
                cliques[cc_i] = np.unique(np.hstack([cc, id_list_ii]))
                add_clique = False
                break
            
        if add_clique:
            cliques.append(id_list_ii)
            
    return cliques

# of the two versions, this is more useful because it ensures more temporal continuity.
def objectness_score_tracks_time(track, vid_score, mean_func=np.nanmean): 
    
    # give an image and a timepoint which the image corresponds to, to get the objectness score. default uses the first timepoint of the video.  
    from skimage.draw import polygon 
    
    img_shape = vid_score[0].shape
    nframes = len(track)
    scores = []
    
    for frame in range(nframes):
        coords = track[frame].copy()
        print(coords)
        if ~np.isnan(coords[0]):
            x1,y1,x2,y2 = coords
            coords = np.vstack([[y1,y1,y2,y2],
                                [x1,x2,x2,x1]])
            rr,cc = polygon(coords[:,0], 
                            coords[:,1], shape=img_shape)
            score = mean_func(vid_score[frame][rr,cc])
            scores.append(score)
        else:
            scores.append(0)
    
    score = np.nanmean(scores)
    
    return score


def track_iou_time(track, use_bbox=True): 
    
    """
    frame to frame iou score 
    """
    nframes = len(track)
    first_diff_iou = []
    
    for frame_no in np.arange(nframes-1):
        
        tra_1 = track[frame_no]
        tra_2 = track[frame_no+1]
        
        if np.isnan(tra_1[0]) or np.isnan(tra_2[0]):
            first_diff_iou.append(np.nan)
        else:
            
            if use_bbox:
                x1, y1, x2, y2 = tra_1
                x1_, y1_, x2_, y2_ = tra_2
                # x1, x2 = np.min(tra_1[...,1]), np.max(tra_1[...,1])
                # y1, y2 = np.min(tra_1[...,0]), np.max(tra_1[...,0])
                
                # x1_, x2_ = np.min(tra_2[...,1]), np.max(tra_2[...,1])
                # y1_, y2_ = np.min(tra_2[...,0]), np.max(tra_2[...,0])
                bbox1 = np.hstack([x1,y1,x2,y2])
                bbox2 = np.hstack([x1_,y1_,x2_,y2_])
                
                # print(bbox1, bbox2)
                iou_12 = get_iou(bbox1, bbox2)
                first_diff_iou.append(iou_12)
    
    first_diff_iou = np.hstack(first_diff_iou)
    
    return first_diff_iou


# the following bypasses the need for point alignment along the tracks? by using area. 
def smoothness_score_tracks_iou(track, mean_func=np.nanmean, second_order=False, use_bbox=True): 
    
    # get the second order track differences
    # from skimage.draw import polygon 
    import numpy as np 
        
    first_diff_iou = track_iou_time(track, use_bbox=use_bbox)
        
    if second_order:
        second_diff_norm = np.gradient(first_diff_iou)
    else:
        second_diff_norm = first_diff_iou.copy()
    # second_diff_norm = np.nanmean(second_diff_norm, axis=1) # obtain the mean over the boundaries., should we use median?

    if second_order:
        score = -mean_func(second_diff_norm) # this should be maxed for iou.
    else:
        score = mean_func(second_diff_norm)
    return score

def nan_stability_score_tracks(track): 
    """
    simply the fraction of frames for which we were able to successfully track the object. 
    """     
    len_track = float(len(track))
    valid_track_times = np.sum(~np.isnan(track[:,0]))

    score = valid_track_times / len_track 
    
    return score


def non_stable_track_suppression_filter(obj_vid, 
                                        org_tracks_list, # use the bbox directly.  
                                        track_overlap_thresh=0.25, 
                                        weight_nan=1., weight_smooth=0.1, max_obj_frames=5,
                                        obj_mean_func=np.nanmean,
                                        smoothness_mean_func=np.nanmean,
                                        fast_comp=True,
                                        debug_viz=False):
    """
    Combines the utility codes above to filter the raw organoid boundary tracks. extends to single and multiple channels.  
        need to absolutely change this... 

        obj_vid: the metric to score objectness .... 
    """
    if debug_viz:
        import pylab as plt 
    
    # 1. apply iou calculations pairwise between tracks to score overlap across channels
    if fast_comp:
        ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks_fast(org_tracks_list)
        # detrend the diagonals.(which is self connections)
        sim_matrix = sim_matrix - np.diag(np.diag(sim_matrix)) 
    else:
        ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks(org_tracks_list) # this concatenates all the tracks etc together, resolving all inter-, intra- overlaps

    sim_matrix_ = sim_matrix.copy()
    sim_matrix_[np.isnan(sim_matrix)] = 0 # replace any nan values which will not be useful. 

    # 2. detect overlaps and cliques (clusters of tracks that correspond to one dominant organoid)
    tracks_overlap = np.where(sim_matrix_ >= track_overlap_thresh)

#    print(tracks_overlap)
    if len(tracks_overlap[0]) > 0:

        # 2b. if there is any evidence of overlap! we collect this all together.  
        overlap_positive_inds = np.vstack(tracks_overlap).T
        overlap_positive_inds = np.sort(overlap_positive_inds, axis=1)
#        print(overlap_positive_inds)
        # overlap_positive_inds = np.unique(overlap_positive_inds, axis=0) #remove duplicate rows.
        overlap_positive_inds = unique_pts(overlap_positive_inds)
        
        # merge these indices to identify unique cliques ... -> as those will likely be trying to track the same organoid. 
        cliq_ids = get_id_cliques(overlap_positive_inds) 
  
        # 3. clique resolution -> determining which organoid is actually being tracked, and which track offers best tracking performance on average from all candidates in the clique. 
        assigned_cliq_track_ids = [] # stores which of the tracks we should use from the overlapped channels.
    
        for cc in cliq_ids[:]:

            # iterate, use objectness score provided by the input vid, to figure out which organoid is being tracked. 
            ind_ids_cc = ind_ids[cc] # what are the possible ids here. 

            obj_score_cc = []
            tra_stable_scores_cc = []
            
            # in the order of organoid id and channel. 
            if debug_viz:
                import seaborn as sns

                ch_colors = sns.color_palette('Set1', len(org_tracks_list))

                fig, ax = plt.subplots()
                ax.imshow(obj_vid[0], alpha=0.5) # just visualise the first frame is enough.
            
            for ind_ids_ccc in ind_ids_cc:
                org_id, org_ch = ind_ids_ccc.split('_')
                org_id = int(org_id)
                org_ch = int(org_ch)
                
                boundary_org = org_tracks_list[org_ch][org_id]

                # this is the problem? 
                # objectness score for deciding the dominant organoid. 
                obj_score = objectness_score_tracks_time(boundary_org[:max_obj_frames], 
                                                          obj_vid[:max_obj_frames,...,org_ch], 
                                                              mean_func=obj_mean_func)
                obj_score_cc.append(obj_score)
                
                # stability score which is weighted on 2 factors. to determine which track. 
                nan_score = nan_stability_score_tracks(boundary_org)
                
                # this should be a minimisation ..... ! 
                # smooth_score = smoothness_score_tracks(boundary_org, 
                #                                       mean_func=np.nanmean)
                smooth_score = smoothness_score_tracks_iou(boundary_org, 
                                                           mean_func=smoothness_mean_func)
                
                tra_stable_scores_cc.append(weight_nan*nan_score+weight_smooth*smooth_score)

            if debug_viz:
                ax.set_title( 'org: %s, stable: %s'  %(ind_ids_cc[np.argmax(obj_score_cc)], 
                                                ind_ids_cc[np.argmax(tra_stable_scores_cc)]))
                plt.show()
            
            # stack all the scores.     
            obj_score_cc = np.hstack(obj_score_cc)
            tra_stable_scores_cc = np.hstack(tra_stable_scores_cc)

            # decide on the organoid and track (argmax)
            cliq_org_id_keep = ind_ids_cc[np.argmax(obj_score_cc)]
            cliq_track_id_keep = ind_ids_cc[np.argmax(tra_stable_scores_cc)]
            
            # save this out for processing. 
            assigned_cliq_track_ids.append([cliq_org_id_keep, cliq_track_id_keep])

        # 4. new org_tracks_list production based on the filtered information. 
        org_tracks_list_out = []

        for list_ii in range(len(org_tracks_list)):

            org_tracks_list_ii = org_tracks_list[list_ii]
            org_tracks_list_ii_out = []

            for org_ii in range(len(org_tracks_list_ii)):
                tra_int_id = str(org_ii)+'_'+str(list_ii) # create the string id lookup. 
                
                include_track = True

                for cliq_ii in range(len(cliq_ids)):
                    ind_ids_cc = ind_ids[cliq_ids[cliq_ii]] # gets the clique members in string form -> is this organoid part of a clique. 
                    
                    if tra_int_id in ind_ids_cc:
                        include_track = False # do not automatically include.  

                        # test is this the dominant organoid in the clique. 
                        cliq_organoid_assign, cliq_organoid_assign_track = assigned_cliq_track_ids[cliq_ii] # get the assignment information of the clique. 
                        
                        if tra_int_id == cliq_organoid_assign:
                            # if this is the dominant organoid then we add the designated track. 
                            org_id_tra_assign, org_ch_tra_assign = cliq_organoid_assign_track.split('_')
                            org_id_tra_assign = int(org_id_tra_assign); org_ch_tra_assign=int(org_ch_tra_assign)
                            
                            org_tracks_list_ii_out.append(org_tracks_list[org_ch_tra_assign][org_id_tra_assign])

                        # do nothing otherwise -> exclude this organoid basically. 
                
                if include_track: 
                    # directly include. 
                    org_tracks_list_ii_out.append(org_tracks_list_ii[org_ii])

            if len(org_tracks_list_ii_out) > 0:
                # stack the tracks. 
                org_tracks_list_ii_out = np.array(org_tracks_list_ii_out)
            
            org_tracks_list_out.append(org_tracks_list_ii_out)
    else:
        org_tracks_list_out = list(org_tracks_list)

    # cleaned tracks, in the same input format as a list of numpy tracks for each channel. 
    return org_tracks_list_out

def non_maxlen_track_suppression_filter(obj_vid, 
                                        org_tracks_list, # use the bbox directly.  
                                        track_overlap_thresh=0.25, 
                                        max_obj_frames=5,
                                        obj_mean_func=np.nanmean,
                                        fast_comp=True,
                                        debug_viz=False):
    """
    Combines the utility codes above to filter the raw organoid boundary tracks. extends to single and multiple channels.  
        need to absolutely change this... 

        obj_vid: the metric to score objectness .... 
    """
    if debug_viz:
        import pylab as plt 
    
    # 1. apply iou calculations pairwise between tracks to score overlap across channels - we only suppress. those that share a lot of overlap. 
    if fast_comp:
        ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks_fast(org_tracks_list)
        # detrend the diagonals.(which is self connections)
        sim_matrix = sim_matrix - np.diag(np.diag(sim_matrix)) 
    else:
        ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks(org_tracks_list) # this concatenates all the tracks etc together, resolving all inter-, intra- overlaps

    sim_matrix_ = sim_matrix.copy()
    sim_matrix_[np.isnan(sim_matrix)] = 0 # replace any nan values which will not be useful. 

    # 2. detect overlaps and cliques (clusters of tracks that correspond to one dominant organoid)
    tracks_overlap = np.where(sim_matrix_ >= track_overlap_thresh)

#    print(tracks_overlap)
    if len(tracks_overlap[0]) > 0:

        # 2b. if there is any evidence of overlap! we collect this all together.  
        overlap_positive_inds = np.vstack(tracks_overlap).T
        overlap_positive_inds = np.sort(overlap_positive_inds, axis=1)
#        print(overlap_positive_inds)
        # overlap_positive_inds = np.unique(overlap_positive_inds, axis=0) #remove duplicate rows.
        overlap_positive_inds = unique_pts(overlap_positive_inds)
        
        # merge these indices to identify unique cliques ... -> as those will likely be trying to track the same organoid. 
        cliq_ids = get_id_cliques(overlap_positive_inds) 
  
        # 3. clique resolution -> determining which organoid is actually being tracked, and which track offers best tracking performance on average from all candidates in the clique. 
        assigned_cliq_track_ids = [] # stores which of the tracks we should use from the overlapped channels.
    
        for cc in cliq_ids[:]:

            # iterate, use objectness score provided by the input vid, to figure out which organoid is being tracked. 
            ind_ids_cc = ind_ids[cc] # what are the possible ids here. 
            print(ind_ids_cc)

            obj_score_cc = []
            tra_stable_scores_cc = []
            
            # in the order of organoid id and channel. 
            if debug_viz:
                import seaborn as sns

                ch_colors = sns.color_palette('Set1', len(org_tracks_list))

                fig, ax = plt.subplots()
                ax.imshow(obj_vid[0], alpha=0.5) # just visualise the first frame is enough.
            
            for ind_ids_ccc in ind_ids_cc:
                org_id, org_ch = ind_ids_ccc.split('_')
                org_id = int(org_id)
                org_ch = int(org_ch)
                
                boundary_org = org_tracks_list[org_ch][org_id]
                print(boundary_org)

                # # this is the problem? 
                # # objectness score for deciding the dominant organoid. 
                # obj_score = [ np.logical_not(org_tra_frame[0]) for org_tra_frame in boundary_org]
                # obj_score = np.sum(obj_score)
                # obj_score_cc.append(obj_score)  # pick the longest track. 
                obj_score = objectness_score_tracks_time(boundary_org[:], 
                                                          obj_vid[...,org_ch], 
                                                          mean_func=obj_mean_func)
                obj_score_cc.append(obj_score)

                # stability score which is weighted on 2 factors. to determine which track. 
                nan_score = nan_stability_score_tracks(boundary_org)
                
                # # this should be a minimisation ..... ! 
                # # smooth_score = smoothness_score_tracks(boundary_org, 
                # #                                       mean_func=np.nanmean)
                # smooth_score = smoothness_score_tracks_iou(boundary_org, 
                #                                            mean_func=smoothness_mean_func)
                
                # tra_stable_scores_cc.append(weight_nan*nan_score+weight_smooth*smooth_score)
                tra_stable_scores_cc.append(nan_score)

            if debug_viz:
                ax.set_title( 'org: %s, stable: %s'  %(ind_ids_cc[np.argmax(obj_score_cc)], 
                                                       ind_ids_cc[np.argmax(obj_score_cc)]))
                plt.show()
            
            # stack all the scores.     
            obj_score_cc = np.hstack(obj_score_cc)
            tra_stable_scores_cc = np.hstack(tra_stable_scores_cc)

            print(obj_score_cc)
            print(tra_stable_scores_cc)
            print('===')

            # decide on the organoid and track (argmax)
            cliq_org_id_keep = ind_ids_cc[np.argmax(obj_score_cc)]
            cliq_track_id_keep = ind_ids_cc[np.argmax(obj_score_cc)]
            
            # save this out for processing. 
            assigned_cliq_track_ids.append([cliq_org_id_keep, cliq_track_id_keep])

        # 4. new org_tracks_list production based on the filtered information. 
        org_tracks_list_out = []

        for list_ii in range(len(org_tracks_list)):

            org_tracks_list_ii = org_tracks_list[list_ii]
            org_tracks_list_ii_out = []

            for org_ii in range(len(org_tracks_list_ii)):
                tra_int_id = str(org_ii)+'_'+str(list_ii) # create the string id lookup. 
                
                include_track = True

                for cliq_ii in range(len(cliq_ids)):
                    ind_ids_cc = ind_ids[cliq_ids[cliq_ii]] # gets the clique members in string form -> is this organoid part of a clique. 
                    
                    if tra_int_id in ind_ids_cc:
                        include_track = False # do not automatically include.  

                        # test is this the dominant organoid in the clique. 
                        cliq_organoid_assign, cliq_organoid_assign_track = assigned_cliq_track_ids[cliq_ii] # get the assignment information of the clique. 
                        
                        if tra_int_id == cliq_organoid_assign:
                            # if this is the dominant organoid then we add the designated track. 
                            org_id_tra_assign, org_ch_tra_assign = cliq_organoid_assign_track.split('_')
                            org_id_tra_assign = int(org_id_tra_assign); org_ch_tra_assign=int(org_ch_tra_assign)
                            
                            org_tracks_list_ii_out.append(org_tracks_list[org_ch_tra_assign][org_id_tra_assign])

                        # do nothing otherwise -> exclude this organoid basically. 
                
                if include_track: 
                    # directly include. 
                    org_tracks_list_ii_out.append(org_tracks_list_ii[org_ii])

            if len(org_tracks_list_ii_out) > 0:
                # stack the tracks. 
                org_tracks_list_ii_out = np.array(org_tracks_list_ii_out)
            
            org_tracks_list_out.append(org_tracks_list_ii_out)
    else:
        org_tracks_list_out = list(org_tracks_list)

    # cleaned tracks, in the same input format as a list of numpy tracks for each channel. 
    return org_tracks_list_out


def filter_nan_tracks( boundary ):
    
    """
    function removes all tracks that for the entire duration is only nan. 
    """
    import numpy as np 
    
    boundaries_out = []
    
    for ii in range(len(boundary)):
        
        tra = boundary[ii]
        tra_len = len(tra)
        n_nans = 0
        
        for tra_tt in tra:
            if np.isnan(tra_tt[0]):
                n_nans+=1
        if n_nans < tra_len:
            boundaries_out.append(tra) # append the whole track
            
    if len(boundaries_out) > 0:
        boundaries_out = np.array(boundaries_out)
        
    return boundaries_out

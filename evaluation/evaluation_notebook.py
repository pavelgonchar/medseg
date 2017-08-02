from medpy import metric
from surface import Surface
import glob
import nibabel as nb
import numpy as np
import os
import os.path as osp

def get_scores(pred,label,vxlspacing):
    volscores = {}

    volscores['dice'] = metric.dc(pred,label)
    volscores['jaccard'] = 0
    # volscores['jaccard'] = metric.binary.jc(pred,label)
    volscores['voe'] = 0
    volscores['rvd'] = 0
    volscores['assd'] = 0
    volscores['msd'] = 0
    # volscores['voe'] = 1. - volscores['jaccard']
    # if np.count_nonzero(label)==0:
    #     volscores['rvd'] = 0
    # else:
    #     volscores['rvd'] = metric.ravd(label,pred)

    # if np.count_nonzero(pred)==0 or np.count_nonzero(label)==0:
    #     volscores['assd'] = 0
    #     volscores['msd'] = 0
    # else:
    #     evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
    #     volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
    #     volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)

    return volscores

def lits_eval(prob, label, outpath):
    loaded_label = nb.load(label)
    loaded_prob = nb.load(prob)
    
    liver_scores = get_scores(loaded_prob.get_data()>=1,loaded_label.get_data()>=1,loaded_label.header.get_zooms()[:3])
    lesion_scores = get_scores(loaded_prob.get_data()==2,loaded_label.get_data()==2,loaded_label.header.get_zooms()[:3])
    print "Liver dice",liver_scores['dice'], "Lesion dice", lesion_scores['dice']
    
    # results.append([label, liver_scores, lesion_scores])

    #create line for csv file
    outstr = str(label) + ','
    for l in [liver_scores, lesion_scores]:
        for k,v in l.iteritems():
            outstr += str(v) + ','
            # outstr += '\n'
    outstr += '\n'

    #create header for csv file if necessary
    if not os.path.isfile(outpath):
        headerstr = 'Volume,'
        for k,v in liver_scores.iteritems():
            headerstr += 'Liver_' + k + ','
        for k,v in liver_scores.iteritems():
            headerstr += 'Lesion_' + k + ','
        headerstr += '\n'
        outstr = headerstr + outstr

    #write to file
    f = open(outpath, 'a+')
    f.write(outstr)
    f.close()

    return label, liver_scores, lesion_scores

# """ Load Labels and Predictions
# """
# root_dir = osp.split(osp.dirname(__file__))[0]
# label_dir = osp.join(root_dir, 'data/lits/Training_Batch')
# prob_dir = osp.join(root_dir, 'output/unet/unet_2d_c3/lits_Training_Batch_val_3D/unet_2d_c3_iter_90000')
# probs = sorted(glob.glob(prob_dir + '/' + 'volume-*_pred.nii'))
# labels = []
# for prob in probs:
#     prob_ind = osp.split(prob)[1].split('_pred')[0].split('-')[1]
#     label = osp.join(label_dir, 'segmentation-{}.nii'.format(prob_ind))
#     labels.append(label)

# """ Loop through all volumes
# """
# results = []
# outpath = osp.join(prob_dir, 'results.csv')
# for label, prob in zip(labels,probs):
#     print label, prob
#     label, liver_scores, lesion_scores = lits_eval(prob, label, outpath)
#     results.append([label, liver_scores, lesion_scores])
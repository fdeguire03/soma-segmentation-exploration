import numpy as np
import matplotlib.pyplot as plt

def s2p_to_masks(stats, iscell):
    cell_stats = stats[iscell]
    masks = np.zeros((240,240,len(cell_stats)))
    for i, stat in enumerate(cell_stats):
        masks[stat['ypix'], stat['xpix'], i] = stat['lam']
    return masks

def normalize(im):
    mini, maxi = im.min(), im.max()
    return (im.astype(float)-mini)/(maxi-mini)
    
def select_middle_frames(scan, return_frames, skip_rows=0, skip_cols=0):
    # Load some frames from the middle of the scan
    num_frames = scan.shape[-1]
    middle_frame = int(np.floor(num_frames / 2))
    frames = slice(max(middle_frame - int(return_frames/2), 0), middle_frame + int(return_frames/2))
    #last_row = -scan.shape[0] if skip_rows == 0 else skip_rows
    #last_col = -scan.shape[1] if skip_cols == 0 else skip_cols
    #mini_scan = scan[skip_rows:-last_row, skip_cols:-last_col, frames]

    return frames

def zoom(im, x_ind=None, y_ind=None, buffer=30):
    if x_ind is None or y_ind is None:
        x_nz, y_nz = np.nonzero(im)
        x_ind = max(0, x_nz.min()-buffer), min(x_nz.max()+buffer, im.shape[0])
        y_ind = max(0, y_nz.min()-buffer), min(y_nz.max()+buffer, im.shape[1])
        return im[x_ind[0]:x_ind[1], y_ind[0]:y_ind[1]], x_ind, y_ind
    return im[x_ind[0]:x_ind[1], y_ind[0]:y_ind[1]]

def smooth_signal(signal, smoothing_factor, norm=2):

    import cvxpy as cp

    reconstructed = cp.Variable(len(signal))
    if norm == 2:
        obj = cp.Minimize(cp.norm2(reconstructed - signal) + smoothing_factor*cp.sum(cp.power(reconstructed[1:] - reconstructed[:-1], 2)))
    else:
        obj = cp.Minimize(cp.norm(reconstructed - signal, 1) + smoothing_factor*cp.norm(reconstructed[1:] - reconstructed[:-1], 2))
    prob = cp.Problem(obj, [])
    prob.solve(solver='CLARABEL')

    return reconstructed.value

    #pip3 install -e /mnt/lab/users/maxgagnon/src/s2p-lbm/suite2p &&\

def plot_comparison_traces(arr1, arr2, assignments, mask_num, labels=('Image 1', 'Image 2'), bins=0, smoothing_factor=0, norm=2):
    
    try:
        old_mask_num = int(assignments[mask_num,0])
        trace1 = normalize(arr1[old_mask_num, :])
        if smoothing_factor > 0:
            trace1 = smooth_signal(trace1, smoothing_factor, norm=norm)
    except ValueError:
        trace1 = np.zeros(len(arr1[0,:]))
        old_mask_num = -1

    try:
        new_mask_num = int(assignments[mask_num,1])
        trace2 = normalize(arr2[new_mask_num, :])
        if smoothing_factor > 0:
            trace2 = smooth_signal(trace2, smoothing_factor, norm=norm)
    except ValueError:
        trace2 = np.zeros(len(arr2[0,:]))
        new_mask_num = -1

    corr_coeff = np.corrcoef(trace1, trace2)[0,1]
    print(f'Correlation coefficient between traces: {corr_coeff}')

    if bins == 0:
        bins = int(min(arr1.shape[-1] / 4, 5000))

    num_plots = len(trace1)//bins
    fig, axes = plt.subplots(num_plots, 1, figsize=(36,int(6*num_plots)))
    plt.title(f'CAIMAN mask {int(old_mask_num)} and Suite2p mask {int(new_mask_num)} (r={round(corr_coeff, 4)})')
    for i in range(num_plots):
        axes[i].plot(np.arange(bins*i, bins*i+bins), trace1[bins*i:bins*i+bins], label=labels[0])
        axes[i].plot(np.arange(bins*i, bins*i+bins), trace2[bins*i:bins*i+bins], label=labels[1])
        axes[i].legend()

def plot_comparison(im1, im2, assignments=None, mask_num=None, labels=('Image 1', 'Image 2'), plot_all=True, buffer=30):

    empty_channel = np.zeros((240,240))
    if mask_num is None:
        fig, axes = plt.subplots(1,3, figsize=(10,10))
        for ax in axes:
            ax.set_axis_off()
        plt.figure(dpi=1200)
        axes[0].imshow(np.dstack((normalize(im1.sum(axis=-1)), empty_channel, empty_channel)))
        axes[0].set_title(labels[0])
        axes[1].imshow(np.dstack((empty_channel, normalize(im2.sum(axis=-1)), empty_channel)))
        axes[1].set_title(labels[1])
        axes[2].imshow(np.dstack((normalize(im1.sum(axis=-1)), normalize(im2.sum(axis=-1)), empty_channel)))
        return

    try:
        old_mask_num = int(assignments[mask_num,0])
    except ValueError:
        old_mask_num = -1
    try:
        new_mask_num = int(assignments[mask_num,1])
    except ValueError:
        new_mask_num = -1
        
    if old_mask_num >= 0:
        _, x_ind, y_ind = zoom(im1[:,:,old_mask_num], buffer=buffer)
    else:
        _, x_ind, y_ind = zoom(im2[:,:,new_mask_num], buffer=buffer)

    fig, axes = plt.subplots(1,3, figsize=(10,10))
    for ax in axes:
        ax.set_axis_off()
    plt.figure(dpi=1200)

    if old_mask_num >= 0:
        axes[0].imshow(np.dstack((zoom(normalize(im1.sum(axis=-1)-im1[:,:,old_mask_num]), x_ind, y_ind), zoom(empty_channel, x_ind, y_ind), zoom(normalize(im1[:,:,old_mask_num]), x_ind, y_ind))))
        axes[0].set_title(labels[0])
    else:
        axes[0].imshow(np.dstack((zoom(normalize(im1.sum(axis=-1)), x_ind, y_ind), zoom(empty_channel, x_ind, y_ind), zoom(empty_channel, x_ind, y_ind))))
        axes[0].set_title('No matching mask')
        axes[2].imshow(np.dstack((zoom(normalize(im1.sum(axis=-1)), x_ind, y_ind), zoom(normalize(im2.sum(axis=-1)), x_ind, y_ind), zoom(normalize(im2[:,:,new_mask_num]), x_ind, y_ind))))
        
    if new_mask_num >= 0:
        axes[1].imshow(np.dstack((zoom(empty_channel, x_ind, y_ind), zoom(normalize(im2.sum(axis=-1)-im2[:,:,new_mask_num]), x_ind, y_ind), zoom(normalize(im2[:,:,new_mask_num]), x_ind, y_ind))))
        axes[1].set_title(labels[1])
    
    else:
        axes[1].imshow(np.dstack((zoom(empty_channel, x_ind, y_ind), zoom(normalize(im2.sum(axis=-1)), x_ind, y_ind), zoom(empty_channel, x_ind, y_ind))))
        axes[1].set_title('No matching mask')
        axes[2].imshow(np.dstack((zoom(normalize(im1.sum(axis=-1)), x_ind, y_ind), zoom(normalize(im2.sum(axis=-1)), x_ind, y_ind), zoom(normalize(im1[:,:,old_mask_num]), x_ind, y_ind))))
    
    if old_mask_num >= 0 and new_mask_num >= 0:
        axes[2].imshow(np.dstack((zoom(normalize(im1.sum(axis=-1)), x_ind, y_ind), zoom(normalize(im2.sum(axis=-1)), x_ind, y_ind), zoom((normalize(im2[:,:,new_mask_num])+normalize(im1[:,:,old_mask_num]))/2, x_ind, y_ind))))

def label_masks(masks):
    fig, ax = plt.subplots(figsize=(14,14))
    ax.imshow(masks.sum(axis=-1))
    maxi = masks.max()
    for i in range(masks.shape[-1]):
        mask = masks[:,:,i]
        x_avg = np.average(np.arange(mask.shape[0]), weights=mask.sum(axis=1))
        y_avg = np.average(np.arange(mask.shape[1]), weights=mask.sum(axis=0))
        t = ax.text(y_avg, x_avg, i, c=str(1/(1+np.exp(0.1*(mask.max() - 3/4*maxi)))), fontsize=10, va='center', ha='center')
        t.set_bbox(dict(facecolor='white', alpha=0.03, linewidth=0))




        
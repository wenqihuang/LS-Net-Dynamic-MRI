from tools.tools import cartesian_mask

if __name__ == "__main__":
    mask_dir = 'mask'
    shape = (18, 192, 192)

    mask_8x = cartesian_mask(shape, 8, sample_n=4, centred=True)
    mask_8x = np.transpose(mask_8x, (1,2,0))

    mask_10x = cartesian_mask(shape, 10, sample_n=4, centred=True)
    mask_10x = np.transpose(mask_10x, (1,2,0))

    mask_12x = cartesian_mask(shape, 12, sample_n=4, centred=True)
    mask_12x = np.transpose(mask_12x, (1,2,0))

    mask_8x_file = os.path.join(mask_dir, 'mask_8x_0.mat')
    mask_10x_file = os.path.join(mask_dir, 'mask_12x_4.mat')
    mask_12x_file = os.path.join(mask_dir, 'mask_12x_4.mat')
    

    datadict = {
        'mask': np.squeeze(mask_8x)
        }
    scio.savemat(mask_8x_file, datadict)

    datadict = {
        'mask': np.squeeze(mask_10x)
        }
    scio.savemat(mask_10x_file, datadict)

    datadict = {
        'mask': np.squeeze(mask_12x)
        }
    scio.savemat(mask_12x_file, datadict)
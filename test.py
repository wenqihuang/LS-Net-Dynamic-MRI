import tensorflow as tf
import os
from model import LplusS_Net, S_Net
from dataset import get_dataset
import argparse
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools.tools import video_summary, mse, tempfft



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', metavar='str', nargs=1, default=['test'], help='training or test')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['16'], help='accelerate rate')
    parser.add_argument('--net', metavar='str', nargs=1, default=['LSNet'], help='L+S Net or S Net')
    parser.add_argument('--weight', metavar='str', nargs=1, default=['models/stable/2020-10-15T16-10-20LSNET_DYNAMIC_V2_MULTICOIL16/epoch-50/ckpt'], help='modeldir in ./models')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['2'], help='GPU No.')
    parser.add_argument('--data', metavar='str', nargs=1, default=['DYNAMIC_V2_MULTICOIL'], help='dataset name')
    parser.add_argument('--learnedSVT', metavar='bool', nargs=1, default=['True'], help='Learned SVT threshold or not')

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    dataset_name = args.data[0].upper()
    mode = args.mode[0]
    batch_size = int(args.batch_size[0])
    niter = int(args.niter[0])
    acc = int(args.acc[0])
    net_name = args.net[0].upper()
    weight_file = args.weight[0]
    learnedSVT = bool(args.learnedSVT[0])

    print('network: ', net_name)
    print('acc: ', acc)
    print('load weight file from: ', weight_file)


    result_dir = os.path.join('results', weight_file.split('/')[2] + net_name + str(acc))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    #logdir = './logs'
    #TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    #summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, TIMESTAMP + net_name + str(acc) + '/'))

    # prepare undersampling mask
    if dataset_name == 'DYNAMIC_V2':
        multi_coil = False
        mask_size = '18_192_192'
    elif dataset_name == 'DYNAMIC_V2_MULTICOIL':
        multi_coil = True
        mask_size = '18_192_192'


    mask_file = 'mask/vista_' + mask_size + '_acc_' + str(acc) + '.mat'
    mask = mat73.loadmat(mask_file)['mask']
    mask = tf.cast(tf.constant(mask), tf.complex64)

    # prepare dataset
    dataset = get_dataset(mode, dataset_name, batch_size, shuffle=False)
    
    # initialize network
    if net_name == 'LSNET':
        net = LplusS_Net(mask, niter, learnedSVT)
    elif net_name == 'SNET':
        net = S_Net(mask, niter)


    net.load_weights(weight_file)
    
    # Iterate over epochs.
    for i, sample in enumerate(dataset):
        # forward
        csm = None
        if multi_coil:
            k0, label, csm = sample
        else:
            k0, label = sample
        label_abs = tf.abs(label)

        k0 = k0 * mask

        if net_name[0:4] == 'SNET':
            t0 = time.time()
            recon = net(k0, csm)
            t1 = time.time()
            recon_abs = tf.abs(recon)
            loss = mse(recon, label)

        else:
            t0 = time.time()
            L_recon, S_recon, LSrecon = net(k0, csm)
            t1 = time.time()
            recon = L_recon + S_recon

            L_recon_abs = tf.abs(L_recon)
            S_recon_abs = tf.abs(S_recon)
            recon_abs = tf.abs(LSrecon)

            loss = mse(LSrecon, label)

        tf.print(i, 'mse =', loss.numpy(), 'time = ', t1-t0)

        result_file = os.path.join(result_dir, 'recon_'+str(i+1)+'.mat')
        if net_name == 'SNET':
            datadict = {
                'recon': np.squeeze(tf.transpose(recon, [0,2,3,1]).numpy())
                }
        else:
            datadict = {
                'recon': np.squeeze(tf.transpose(LSrecon, [0,2,3,1]).numpy()), 
                'L':np.squeeze(tf.transpose(L_recon, [0,2,3,1]).numpy()), 
                'S':np.squeeze(tf.transpose(S_recon, [0,2,3,1]).numpy())
                }
        scio.savemat(result_file, datadict)

        # record gif
        # with summary_writer.as_default():
        #     if net_name[0:4] == 'SNET':
        #         combine_video = tf.concat([label_abs[0:1,:,:,:], recon_abs[0:1,:,:,:]], axis=0).numpy()
        #     else:
        #         combine_video = tf.concat([label_abs[0:1,:,:,:], recon_abs[0:1,:,:,:], L_recon_abs[0:1,:,:,:], S_recon_abs[0:1,:,:,:]], axis=0).numpy()
        #     combine_video = np.expand_dims(combine_video, -1)
        #     video_summary('convin-'+str(i+1), combine_video, step=1, fps=10)



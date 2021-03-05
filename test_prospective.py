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
    parser.add_argument('--acc', metavar='int', nargs=1, default=['4'], help='accelerate rate')
    parser.add_argument('--net', metavar='str', nargs=1, default=['LSNet'], help='L+S Net or S Net')
    parser.add_argument('--weight', metavar='str', nargs=1, default=['models/stable/2020-09-02T11-38-47LSNET_DYNAMIC_V28_learnSVT/epoch-50/ckpt'], help='modeldir in ./models')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['6'], help='GPU No.')
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


    result_dir = os.path.join('results/prospective', weight_file.split('/')[2] + net_name + str(acc))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    
    for i in range(2,8):
        k0 = mat73.loadmat('data/prospective/ku'+str(i)+'.mat')['ku']
        csm = mat73.loadmat('data/prospective/csm'+str(i)+'.mat')['csm']

        k0 = tf.convert_to_tensor(k0, dtype=tf.complex64)
        csm = tf.convert_to_tensor(csm, dtype=tf.complex64)
        csm = tf.expand_dims(csm, 2)

        k0 = tf.expand_dims(k0, 0) #batch
        csm = tf.expand_dims(csm, 0) #batch

        k0 = tf.transpose(k0, [0,4,3,1,2])
        csm = tf.transpose(csm, [0,4,3,1,2])

        mask = tf.cast(tf.abs(k0) > 0, tf.complex64)

        # initialize network
        net = LplusS_Net(mask, niter, learnedSVT)
        net.load_weights(weight_file)
        

        t0 = time.time()
        L_recon, S_recon, LSrecon = net(k0, csm)
        t1 = time.time()
        recon = L_recon + S_recon

        L_recon_abs = tf.abs(L_recon)
        S_recon_abs = tf.abs(S_recon)
        recon_abs = tf.abs(LSrecon)

        result_file = os.path.join(result_dir, 'recon_'+str(i)+'.mat')
        datadict = {
            'recon': np.squeeze(tf.transpose(LSrecon, [0,2,3,1]).numpy()), 
            'L':np.squeeze(tf.transpose(L_recon, [0,2,3,1]).numpy()), 
            'S':np.squeeze(tf.transpose(S_recon, [0,2,3,1]).numpy())
            }
        scio.savemat(result_file, datadict)



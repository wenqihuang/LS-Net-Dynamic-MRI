# python main.py --net='LSNET' --acc=16 --data='DYNAMIC_V2_MULTICOIL' --gpu=2 --batch_size=1 --learnedSVT=True
# python main.py --net='SNET' --acc=16 --data='DYNAMIC_V2_MULTICOIL' --gpu=2 --batch_size=1
import os
import argparse
import tensorflow as tf
from model import LplusS_Net, S_Net
from dataset import get_dataset
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools.tools import video_summary

from tools.tools import tempfft, mse


#tf.debugging.set_log_device_placement(True)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['16'], help='accelerate rate')
    parser.add_argument('--net', metavar='str', nargs=1, default=['LSNET'], help='L+S Net or S Net')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--data', metavar='str', nargs=1, default=['DYNAMIC_V2_MULTICOIL'], help='dataset name, \
                        DYNAMIC_V2_MULTICOIL for multi-coil dataset, DYNAMIC_V2 for single-coil dataset.')
    parser.add_argument('--learnedSVT', metavar='bool', nargs=1, default=['True'], help='Learned SVT threshold or not')
    

    args = parser.parse_args()
    
    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)
    
    mode = 'training'
    dataset_name = args.data[0].upper()
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])

    acc = int(args.acc[0])
    net_name = args.net[0].upper()
    niter = int(args.niter[0])
    learnedSVT = bool(args.learnedSVT[0])


    logdir = './logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    model_id  = TIMESTAMP + net_name + '_' + dataset_name + str(acc)
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, model_id + '/'))

    modeldir = os.path.join('models/', model_id)
    os.makedirs(modeldir)

    # prepare undersampling mask
    if dataset_name == 'DYNAMIC_V2':
        multi_coil = False
        mask_size = '18_192_192'
    elif dataset_name == 'DYNAMIC_V2_MULTICOIL':
        multi_coil = True
        mask_size = '18_192_192'

    if dataset_name != 'DUMMY':
        mask_file = 'mask/vista_' + mask_size + '_acc_' + str(acc) + '.mat'
        mask = mat73.loadmat(mask_file)['mask']
    else:
        multi_coil = True
        mask_file = 'mask/dummy_mask_8.mat'
        mask = scio.loadmat(mask_file)['mask']
    mask = tf.cast(tf.constant(mask), tf.complex64)

    # prepare dataset
    dataset = get_dataset(mode, dataset_name, batch_size, shuffle=True)
    tf.print('dataset loaded.')

    # initialize network
    if net_name == 'LSNET':
        net = LplusS_Net(mask, niter, learnedSVT)
    elif net_name == 'SNET':
        net = S_Net(mask, niter)

    tf.print('network initialized.')

    learning_rate_org = learning_rate
    learning_rate_decay = 0.95

    optimizer = tf.optimizers.Adam(learning_rate_org)
    
    
    # Iterate over epochs.
    total_step = 0
    param_num = 0
    loss = 0

    for epoch in range(num_epoch):
        for step, sample in enumerate(dataset):
            
            # forward
            t0 = time.time()
            csm = None
            with tf.GradientTape() as tape:
                if multi_coil:
                    k0, label, csm = sample
                    if k0 == None:
                        continue
                else:
                    k0, label = sample
                if k0.shape[0] < batch_size:
                    continue

                label_abs = tf.abs(label)

                max_val = tf.reduce_max(label_abs)
                k0 /= tf.complex(max_val, 0.0)
                label /= tf.complex(max_val, 0.0)
                label_abs = tf.abs(label)

                k0 = k0 * mask

                if net_name == 'SNET':
                    recon = net(k0, csm)

                    recon_abs = tf.abs(recon)
                    loss_mse = mse(recon, label)
                else:
                    L_recon, S_recon, LSrecon = net(k0, csm)
                    recon = L_recon + S_recon

                    L_recon_abs = tf.abs(L_recon)
                    S_recon_abs = tf.abs(S_recon)
                    recon_abs = tf.abs(LSrecon)
                    loss_mse = mse(recon, label)

                loss = loss_mse #mse ok


            # backward
            grads = tape.gradient(loss, net.trainable_weights)
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            # record loss
            with summary_writer.as_default():
                tf.summary.scalar('loss/total', loss_mse.numpy(), step=total_step)

            # record gif
            
            if step % 20 == 0:
                with summary_writer.as_default():
                    if net_name == 'SNET':
                        combine_video = tf.concat([label_abs[0:1,:,:,:], recon_abs[0:1,:,:,:]], axis=0).numpy()
                    else:
                        combine_video = tf.concat([label_abs[0:1,:,:,:], recon_abs[0:1,:,:,:], \
                                                L_recon_abs[0:1,:,:,:], S_recon_abs[0:1,:,:,:]], axis=0).numpy()
                    combine_video = np.expand_dims(combine_video, -1)
                    video_summary('result', combine_video, step=total_step, fps=10)
            
            # calculate parameter number
            if total_step == 0:
                param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])

            # log output
            tf.print('Epoch', epoch+1, '/', num_epoch, 'Step', step, 'loss =', loss.numpy(), 
                    'time', time.time() - t0, 'lr = ', learning_rate, 'param_num', param_num)
            total_step += 1

        # learning rate decay for each epoch
        learning_rate = learning_rate_org * learning_rate_decay ** (epoch + 1)
        optimizer = tf.optimizers.Adam(learning_rate)

        # save model each epoch
        if (epoch+1) % 10 == 0:
            model_epoch_dir = os.path.join(modeldir,'epoch-'+str(epoch+1), 'ckpt')
            net.save_weights(model_epoch_dir, save_format='tf')


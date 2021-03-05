import tensorflow as tf
import os
import scipy.io as scio
import numpy as np
import time
import tools.mymath as mymath
import mat73 # 注意，如果.mat文件版本大于等于7.3，要用mat73读 否则用scio读

# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPUs = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(GPUs[0], True)

def Generate_data(mode, data_num):
    time_start = time.time()
    #sep_data_root = '/data0/wenqi/Dynamic SSFP v2/5_cropped_and_argemented/'+mode+' data/seperate_5_coil_aug' # 各个通道重建好的图像
    #com_data_root = '/data0/wenqi/Dynamic SSFP v2/5_cropped_and_argemented/'+mode+' data/combined_5_coil_aug' # 各通道看重建好的图像再adaptivecombine的单通道图像
    #csm_data_root = '/data0/wenqi/Dynamic SSFP v2/5_cropped_and_argemented/'+mode+' data/csm'

    sep_data_root = '/data0/wenqi/Dynamic SSFP v2/20_coil/'+mode+' data/seperate_multi_coil_aug' # 各个通道重建好的图像
    com_data_root = '/data0/wenqi/Dynamic SSFP v2/20_coil/'+mode+' data/combined_multi_coil_aug' # 各通道看重建好的图像再adaptivecombine的单通道图像
    csm_data_root = '/data0/wenqi/Dynamic SSFP v2/20_coil/'+mode+' data/csm'

    tfrecordfile = 'data/20coil/'+'cine_multicoil_'+mode+'.tfrecord'
    #option = tf.io.TFRecordOptions(compression_type="ZLIB", compression_level=9)
    #option = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    writer = tf.io.TFRecordWriter(tfrecordfile)
    
    for i in range(data_num):
        tf.print('Processing ',i+1,'/',data_num)
        # generate kspace data
        sep_filename = os.path.join(sep_data_root, 'label'+str(i+1)+'.mat')
        subdata = mat73.loadmat(sep_filename)
        sep_label = subdata['sublabel']  # 192, 192, 18, 5
        sep_label = np.transpose(sep_label, (3, 2, 0, 1))  #nc, nt, nx, ny
        
        k = mymath.fft2c(sep_label)
        k = k.astype(np.complex64)

        # generate label data
        com_filename = os.path.join(com_data_root, 'label'+str(i+1)+'.mat')
        subdata = mat73.loadmat(com_filename)
        # subdata = scio.loadmat(com_filename) #注意，如果.mat文件版本大于等于7.3，要用mat73读 否则用scio读
        label = subdata['sublabel']  # 192, 192, 18
        label = np.transpose(label, (2, 0, 1))  #nt, nx, ny

        # generate csm data
        csm_filename = os.path.join(csm_data_root, 'csm'+str(i+1)+'.mat')
        subdata = mat73.loadmat(csm_filename) 
        # subdata = scio.loadmat(csm_filename) #注意，如果.mat文件版本大于等于7.3，要用mat73读 否则用scio读
        csm = subdata['csm']  # 192, 192, 18, 5
        csm = np.expand_dims(csm, 2)
        csm = np.transpose(csm, (3, 2, 0, 1))  #nc, nt, nx, ny

        # split real and imag
        k_real = np.real(k)
        k_imag = np.imag(k)
        label_real = np.real(label)
        label_imag = np.imag(label)
        csm_real = np.real(csm)
        csm_imag = np.imag(csm)

        # write into tfrecord
        features = {}
        features['k_real'] = tf.train.Feature(float_list = tf.train.FloatList(value=k_real.reshape(-1)))
        features['k_imag'] = tf.train.Feature(float_list = tf.train.FloatList(value=k_imag.reshape(-1)))
        features['label_real'] = tf.train.Feature(float_list = tf.train.FloatList(value=label_real.reshape(-1)))
        features['label_imag'] = tf.train.Feature(float_list = tf.train.FloatList(value=label_imag.reshape(-1)))
        features['csm_real'] = tf.train.Feature(float_list = tf.train.FloatList(value=csm_real.reshape(-1)))
        features['csm_imag'] = tf.train.Feature(float_list = tf.train.FloatList(value=csm_imag.reshape(-1)))
        features['k_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=k_real.shape))
        features['img_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=label_real.shape))
        features['csm_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=csm_real.shape))

        tf_features = tf.train.Features(feature= features)        
        tf_example = tf.train.Example(features = tf_features)
        tf_serialized = tf_example.SerializeToString()
        
        writer.write(tf_serialized)
    writer.close()

    time_stop = time.time()
    tf.print('Done! Time used: ', time_stop-time_start)

if __name__ == "__main__":
    Generate_data('training', 800)
    Generate_data('test', 118)
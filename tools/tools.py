import tempfile
import os
import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.io as scio

def video_summary(name, video, step=None, fps=10):
    name = tf.constant(name).numpy().decode('utf-8')
    video = np.array(video)
    if video.dtype in (np.float32, np.float64):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf.compat.v1.Summary()
        image = tf.compat.v1.Summary.Image(
            height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask

def tempfft(input, inv):
    if len(input.shape) == 4:
        nb, nt, nx, ny = np.float32(input.shape)
        nt = tf.constant(np.complex64(nt + 0j))

        if inv:
            x = tf.transpose(input, perm=[0,2,3,1])
            #x = tf.signal.fftshift(x, 3)
            x = tf.signal.ifft(x)
            x = tf.transpose(x, perm=[0,3,1,2])
            x = x * tf.sqrt(nt)
        else:
            x = tf.transpose(input, perm=[0,2,3,1])
            x = tf.signal.fft(x)
            #x = tf.signal.fftshift(x, 3)
            x = tf.transpose(x, perm=[0,3,1,2])
            x = x / tf.sqrt(nt)
    else:
        nb, nt, nx, ny, _ = np.float32(input.shape)
        nt = tf.constant(np.complex64(nt + 0j))

        if inv:
            x = tf.transpose(input, perm=[0,2,3,4,1])
            #x = tf.signal.fftshift(x, 4)
            x = tf.signal.ifft(x)
            x = tf.transpose(x, perm=[0,4,1,2,3])
            x = x * tf.sqrt(nt)
        else:
            x = tf.transpose(input, perm=[0,2,3,4,1])
            x = tf.signal.fft(x)
            #x = tf.signal.fftshift(x, 4)
            x = tf.transpose(x, perm=[0,4,1,2,3])
            x = x / tf.sqrt(nt)
    return x


def mse(recon, label):
    if recon.dtype == tf.complex64:
        residual_cplx = recon - label
        residual = tf.stack([tf.math.real(residual_cplx), tf.math.imag(residual_cplx)], axis=-1)
        mse = tf.reduce_mean(residual**2)
    else:
        residual = recon - label
        mse = tf.reduce_mean(residual**2)
    return mse
    

def fft2c_mri(x):
    # nb nx ny nt
    X = tf.signal.fftshift(x, 2)
    X = tf.transpose(X, perm=[0,1,3,2]) # permute to make nx dimension the last one.
    X = tf.signal.fft(X)
    X = tf.transpose(X, perm=[0,1,3,2]) # permute back to original order.
    nb, nt, nx, ny = np.float32(x.shape)
    nx = tf.constant(np.complex64(nx + 0j))
    ny = tf.constant(np.complex64(ny + 0j))
    X = tf.signal.fftshift(X, 2) / tf.sqrt(nx)
    X = tf.signal.fftshift(X, 3)
    X = tf.signal.fft(X)
    X = tf.signal.fftshift(X, 3) / tf.sqrt(ny)
        
    return X

def ifft2c_mri(X):
    # nb nx ny nt
    x = tf.signal.fftshift(X, 2)
    x = tf.transpose(x, perm=[0,1,3,2]) # permute a to make nx dimension the last one.
    x = tf.signal.ifft(x)
    x = tf.transpose(x, perm=[0,1,3,2]) # permute back to original order.
    nb, nt, nx, ny = np.float32(X.shape)
    nx = tf.constant(np.complex64(nx + 0j))
    ny = tf.constant(np.complex64(ny + 0j))

    x = tf.signal.fftshift(x, 2) * tf.sqrt(nx)

    x = tf.signal.fftshift(x, 3)
    x = tf.signal.ifft(x)
    x = tf.signal.fftshift(x, 3) * tf.sqrt(ny)
        
    return x
    

def sos(x):
    # x: nb, ncoil, nt, nx, ny; complex64
    x = tf.math.reduce_sum(tf.abs(x**2), axis=1)
    x = x**(1.0/2)
    return x
    

def softthres(x, thres):
    x_abs = tf.abs(x)
    coef = tf.nn.relu(x_abs - thres) / (x_abs + 1e-10)
    coef = tf.cast(coef, tf.complex64)
    return coef * x

class Emat_xyt():
    def __init__(self, mask):
        super(Emat_xyt, self).__init__()
        self.mask = mask

    def mtimes(self, b, inv, csm):
        if csm == None:
            if inv:
                x = self._ifft2c_mri_singlecoil(b * self.mask)
            else:
                x = self._fft2c_mri_singlecoil(b) * self.mask
        else:
            if len(self.mask.shape) > 3:
                if inv:
                    x = self._ifft2c_mri_multicoil(b * self.mask[:,:,0:b.shape[2],:,:])
                    x = x * tf.math.conj(csm)
                    x = tf.reduce_sum(x, 1) #/ tf.cast(tf.reduce_sum(tf.abs(csm)**2, 1), dtype=tf.complex64)
                else:
                    b = tf.expand_dims(b, 1) * csm
                    x = self._fft2c_mri_multicoil(b) * self.mask[:,:,0:b.shape[2],:,:]
            else:
                if inv:
                    x = self._ifft2c_mri_multicoil(b * self.mask)
                    x = x * tf.math.conj(csm)
                    x = tf.reduce_sum(x, 1) #/ tf.cast(tf.reduce_sum(tf.abs(csm)**2, 1), dtype=tf.complex64)
                else:
                    b = tf.expand_dims(b, 1) * csm
                    x = self._fft2c_mri_multicoil(b) * self.mask
        
        return x
    
    def _fft2c_mri_multicoil(self, x):
        # nb nt nx ny -> nb, nc, nt, nx, ny
        X = tf.signal.fftshift(x, 3)
        X = tf.transpose(X, perm=[0,1,2,4,3]) # permute to make nx dimension the last one.
        X = tf.signal.fft(X)
        X = tf.transpose(X, perm=[0,1,2,4,3]) # permute back to original order.
        nb, nc, nt, nx, ny = np.float32(x.shape)
        nx = tf.constant(np.complex64(nx + 0j))
        ny = tf.constant(np.complex64(ny + 0j))
        X = tf.signal.fftshift(X, 3) / tf.sqrt(nx)
        X = tf.signal.fftshift(X, 4)
        X = tf.signal.fft(X)
        X = tf.signal.fftshift(X, 4) / tf.sqrt(ny)
        
        return X

    def _ifft2c_mri_multicoil(self, X):
        # nb nt nx ny -> nb, nc, nt, nx, ny
        x = tf.signal.fftshift(X, 3)
        x = tf.transpose(x, perm=[0,1,2,4,3]) # permute a to make nx dimension the last one.
        x = tf.signal.ifft(x)
        x = tf.transpose(x, perm=[0,1,2,4,3]) # permute back to original order.
        nb, nc, nt, nx, ny = np.float32(X.shape)
        nx = tf.constant(np.complex64(nx + 0j))
        ny = tf.constant(np.complex64(ny + 0j))

        x = tf.signal.fftshift(x, 3) * tf.sqrt(nx)

        x = tf.signal.fftshift(x, 4)
        x = tf.signal.ifft(x)
        x = tf.signal.fftshift(x, 4) * tf.sqrt(ny)
        
        return x

    def _fft2c_mri_singlecoil(self, x):
        # nb nx ny nt
        X = tf.signal.fftshift(x, 2)
        X = tf.transpose(X, perm=[0,1,3,2]) # permute to make nx dimension the last one.
        X = tf.signal.fft(X)
        X = tf.transpose(X, perm=[0,1,3,2]) # permute back to original order.
        nb, nt, nx, ny = np.float32(x.shape)
        nx = tf.constant(np.complex64(nx + 0j))
        ny = tf.constant(np.complex64(ny + 0j))
        X = tf.signal.fftshift(X, 2) / tf.sqrt(nx)
        X = tf.signal.fftshift(X, 3)
        X = tf.signal.fft(X)
        X = tf.signal.fftshift(X, 3) / tf.sqrt(ny)
        
        return X

    def _ifft2c_mri_singlecoil(self, X):
        # nb nx ny nt
        x = tf.signal.fftshift(X, 2)
        x = tf.transpose(x, perm=[0,1,3,2]) # permute a to make nx dimension the last one.
        x = tf.signal.ifft(x)
        x = tf.transpose(x, perm=[0,1,3,2]) # permute back to original order.
        nb, nt, nx, ny = np.float32(X.shape)
        nx = tf.constant(np.complex64(nx + 0j))
        ny = tf.constant(np.complex64(ny + 0j))

        x = tf.signal.fftshift(x, 2) * tf.sqrt(nx)

        x = tf.signal.fftshift(x, 3)
        x = tf.signal.ifft(x)
        x = tf.signal.fftshift(x, 3) * tf.sqrt(ny)
        
        return x

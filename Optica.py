from math import pi
import numpy as np
import os
import matplotlib.pyplot as plt
import wget
import cv2


def get_obj(file_path='./obj.png', size=(1000, 1000)):
    if os.path.exists(file_path):
        pass
    else:
        html = "https://www1.cs.columbia.edu/CAVE/databases/multispectral/images/feathers.png"
        wget.download(html, file_path)
    im = cv2.imread(file_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, size)
    return im


def gen_r(width, height, res):
    meshu = np.mgrid[-width / 2:width / 2:res, -height / 2:height / 2:res]
    x = meshu[0]
    y = meshu[1]
    r = (x ** 2 + y ** 2) ** 0.5
    return r


class DOE:
    def __init__(self, width=0.1, height=0.1, minlam=4 * 10 ** -5, maxlam=8 * 10 ** -5,
                 minh=0.03, materail=None, f=5, res=0.0001):
        if materail is None:
            SiO2_nk_data = np.loadtxt('E:/D/study/FDTDNN/mytmm/SiO2_index.csv', encoding='UTF-8-sig', delimiter=',')
            SiO2_nk_data[:, 0] = SiO2_nk_data[:, 0] * 10 ** -4
            materail = SiO2_nk_data
        self.width = width
        self.height = height
        self.res = res
        self.minlam = minlam
        self.maxlam = maxlam
        self.minh = minh
        self.materail = materail
        self.f = f

    def generate_doe(self):
        numw = int(self.width / self.res)
        numl = int(self.height / self.res)
        hdoe = np.ones([numw, numl]) * self.minh
        mesh = np.mgrid[-self.width / 2:self.width / 2:self.res, -self.height / 2:self.height / 2:self.res]
        x = mesh[0] + 10 ** -20
        y = mesh[1] + 10 ** -20
        r = (x ** 2 + y ** 2) ** 0.5
        theta = np.arctan(y / x) + pi / 2
        theta[0:int(numw / 2), :] += pi
        hdoe += self.caculate_h(r, theta, -1)
        return hdoe

    def caculate_n(self, lam):
        n = np.interp(lam, self.materail[:, 0].real, self.materail[:, 1])
        return n

    def caculate_lam(self, theta, wing=3):
        theta = theta % ((2 * pi) / wing)
        lam = self.minlam + (self.maxlam - self.minlam) * wing * theta / (2 * pi)
        return lam

    def caculate_h(self, r, theta, k):
        lam = self.caculate_lam(theta)
        n = self.caculate_n(lam)
        deltah = ((k * lam) - ((r ** 2 + self.f ** 2) ** 0.5 - self.f)) / (n - 1)
        deltah = deltah % (lam/(n-1))
        return deltah


class Diffraction:
    def __init__(self, u0=None, d=5,  lam=0.6):
        if u0 is None:
            u0 = [0.1, 0.1, 0.0001, 1]
            h = np.zeros([1000, 1000])
            h[490:510, 490:510] = 1
            u0 = u0.append(h)
        self.u0 = u0
        self.nsample = u0[0]/u0[2]
        self.hdoe = u0[4]
        self.d = d
        self.lam = lam
        self.k = 2*pi/lam

    def sfft(self):
        r0 = gen_r(self.u0[0], self.u0[1], self.u0[2])
        lu = self.nsample * self.lam * self.d / self.u0[0]
        print('size of observe surface is %.f cm' % lu)
        r = gen_r(lu, lu, lu/self.nsample)
        doe = DOE()
        n = doe.caculate_n(self.lam)
        phi0 = self.u0[3]*2*pi/self.lam*(n-1)*self.hdoe
        fft = np.fft.fftshift(np.fft.fft2(np.exp(1j*phi0) * np.exp(1j*self.k/(2*self.d)*r0**2)))
        u = np.exp(self.d*self.k*1j)*np.exp(1j*self.k*r**2/2/self.d)*fft/(1j*self.d*self.lam)
        return u

    def dfft(self):
        doe = DOE()
        n = doe.caculate_n(self.lam)
        phi0 = self.u0[3]*2*pi/self.lam*(n-1)*self.hdoe
        fft = np.fft.fftshift(np.fft.fft2(np.exp(1j * phi0)))
        luv = self.nsample/2/self.u0[0]
        print('size of frequency surface is %.f cm^-1' % (2*luv))
        ruv = gen_r(2 * luv, 2 * luv, 2 * luv / self.nsample)
        u = np.fft.ifft2(fft * np.exp(1j * self.k * self.d * (1 - self.lam**2 * ruv**2)**0.5))
        return u


if __name__ == '__main__':
    SiO2_nk_data = np.loadtxt('E:/D/study/FDTDNN/mytmm/SiO2_index.csv', encoding='UTF-8-sig', delimiter=',')
    SiO2_nk_data[:, 0] = SiO2_nk_data[:, 0] * 10**-4
    A = DOE(width=0.1, height=0.1, minlam=4 * 10 ** -5, maxlam=8 * 10 ** -5,
            minh=0.03, materail=SiO2_nk_data, f=5, res=0.0001)
    hdoe = A.generate_doe()  # cm
    obj_path = './obj.png'
    obj = get_obj(obj_path, (1000, 1000))
    u0 = [0.1, 0.1, 0.0001, 1, hdoe]
    diffraction = Diffraction(u0=u0, d=5, lam=6*10**-5)
    u = diffraction.dfft()
    i = u * np.conj(u)
    i = i/np.max(i)
    i = i.real
    plt.figure()
    plt.imshow(i, cmap='gray')
    plt.show()

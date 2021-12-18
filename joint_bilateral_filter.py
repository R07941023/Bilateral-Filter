import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


class Joint_bilateral_filter( object ):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):

        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def integral_image(self, img):
        img_integral = []
        if img.ndim == 3:
            for i in range( img.shape[2] ):
                new_mask = np.zeros( (img.shape[0], img.shape[1]) )
                mask = img[:, :, i]
                for j1 in range( img.shape[0] ):
                    for j2 in range( img.shape[1] ):
                        if j1 == 0 and j2 == 0:
                            new_mask[j1][j2] = mask[j1][j2]
                        elif j1 == 0:
                            new_mask[j1][j2] = new_mask[j1][j2 - 1] + mask[j1][j2]
                        elif j2 == 0:
                            new_mask[j1][j2] = new_mask[j1 - 1][j2] + mask[j1][j2]
                        else:
                            new_mask[j1][j2] = new_mask[j1 - 1][j2] + new_mask[j1][j2 - 1] - new_mask[j1 - 1][j2 - 1] + \
                                               mask[j1, j2]
                img_integral.append( new_mask )
        else:
            new_mask = np.zeros( (img.shape[0], img.shape[1]) )
            mask = img[:, :]
            for j1 in range( img.shape[0] ):
                for j2 in range( img.shape[1] ):
                    if j1 == 0 and j2 == 0:
                        new_mask[j1][j2] = mask[j1][j2]
                    elif j1 == 0:
                        new_mask[j1][j2] = new_mask[j1][j2 - 1] + mask[j1][j2]
                    elif j2 == 0:
                        new_mask[j1][j2] = new_mask[j1 - 1][j2] + mask[j1][j2]
                    else:
                        new_mask[j1][j2] = new_mask[j1 - 1][j2] + new_mask[j1][j2 - 1] - new_mask[j1 - 1][j2 - 1] + \
                                           mask[j1, j2]
            img_integral.append( new_mask )

        img_integral = np.array( img_integral )

        return img_integral

    def build_kernel(self, sigma, winsiz):
        x, y = np.mgrid[-(winsiz-1)/2:(winsiz-1)/2+1, -(winsiz-1)/2:(winsiz-1)/2+1]
        print(x)
        print(sigma, winsiz)
        print(x.shape)
        exit()
        kernel = np.exp(-(x**2+y**2)/(2*sigma**2))  # /(2*math.pi*sigma**2)
        kernel = kernel/kernel.sum()
        return kernel

    def bilateral_filter(self, input, guidance, kernel_s, sigma_r):
        ## TODO
        boundary = int((kernel_s.shape[0]-1)/2)

        # normalize
        guidance = guidance.astype(np.float64)/255.
        input = input.astype(np.float64)/255.
        kernel_s = kernel_s.astype(np.float64)
        padding_input = cv2.copyMakeBorder( input, boundary, boundary, boundary, boundary, cv2.BORDER_REFLECT ).astype(np.float64)
        padding_guidance = cv2.copyMakeBorder( guidance, boundary, boundary, boundary, boundary, cv2.BORDER_REFLECT ).astype(np.float64)
        # bilateral_filter
        output = np.zeros( input.shape ).astype(np.float64)
        if input.ndim == 3 and guidance.ndim == 3:  # the channel number > 1
            for i in range(input.shape[2]):  # channel
                for j in range(input.shape[0]):
                    for k in range( input.shape[1] ):
                        # kernel
                        Tpr = padding_guidance[j:j+kernel_s.shape[0], k:k+kernel_s.shape[0], 0]
                        Tqr = guidance[j][k][0]
                        Tpg = padding_guidance[j:j + kernel_s.shape[0], k:k + kernel_s.shape[0], 1]
                        Tqg = guidance[j][k][1]
                        Tpb = padding_guidance[j:j + kernel_s.shape[0], k:k + kernel_s.shape[0], 2]
                        Tqb = guidance[j][k][2]
                        kernel_r = np.exp(-((Tpr - Tqr)**2+(Tpg - Tqg)**2+(Tpb - Tqb)**2)/(2*sigma_r**2))
                        kernel = np.multiply(kernel_r, kernel_s)
                        kernel = kernel/kernel.sum()
                        # mapping
                        map = padding_input[j:j+kernel_s.shape[0], k:k+kernel_s.shape[0], i]
                        mapping = np.multiply( kernel, map )
                        temp_unit = mapping.sum()*255.
                        output[j][k][i] = temp_unit.astype(np.float64)

        if input.ndim == 3 and guidance.ndim == 2:  # the channel number > 1
            for i in range(input.shape[2]):  # channel
                for j in range(input.shape[0]):
                    for k in range( input.shape[1] ):
                        map = padding_input[j:j+kernel_s.shape[0], k:k+kernel_s.shape[0], i]
                        Tp = padding_guidance[j:j+kernel_s.shape[0], k:k+kernel_s.shape[0]]
                        Tq = guidance[j][k]
                        kernel_r = np.exp(-(Tp - Tq)**2/(2*sigma_r**2))
                        kernel = np.multiply(kernel_r, kernel_s)
                        kernel = kernel/kernel.sum()
                        mapping = np.multiply( kernel, map )
                        temp_unit = mapping.sum() * 255.
                        output[j][k][i] = temp_unit.astype(np.float64)

        elif input.ndim == 2 and guidance.ndim == 2:
            for j in range(input.shape[0]):
                for k in range( input.shape[1] ):
                    map = padding_input[j:j+kernel_s.shape[0], k:k+kernel_s.shape[0]]
                    Tp = padding_guidance[j:j+kernel_s.shape[0], k:k+kernel_s.shape[0]]
                    Tq = guidance[j][k]
                    kernel_r = np.exp(-(Tp - Tq)**2/(2*sigma_r**2))
                    kernel = np.multiply(kernel_r, kernel_s)
                    kernel = kernel/kernel.sum()
                    mapping = np.multiply( kernel, map )
                    temp_unit = mapping.sum() * 255.
                    output[j][k][i] = temp_unit.astype(np.float64)

        # output = output.astype( 'uint8' )

        return output

    def joint_bilateral_filter(self, input, guidance):
        ti = time.time()
        # input_integral = self.integral_image( input )
        kernel_s = self.build_kernel(self.sigma_s, 3*self.sigma_s*2+1)
        # print(kernel_s[0])
        # exit()
        jbf = self.bilateral_filter(input, guidance, kernel_s, self.sigma_r)
        tf = time.time()
        print('t = ', tf-ti, 's')
        return jbf

    def hw1(self, input, ):
        input = input.astype( 'float64' )

        sigma_s_set = [1, 2, 3]
        sigma_r_set = [0.05, 0.1, 0.2]

        for sigma_s in sigma_s_set:
            for sigma_r in sigma_r_set:
                print('sigma_s/sigma_r = ', sigma_s, '/', sigma_r)
                w = [10, 0, 0]  # RGB
                n = 0
                set_r, set_g, set_b, dir_loss = [], [], [], {}
                kernel_s = self.build_kernel( sigma_s, 3*sigma_s*2+1 )

                jbf_gt = self.bilateral_filter( input.astype( 'uint8' ), input.astype( 'uint8' ), kernel_s, sigma_r )
                # jbf_gt = self.bilateral_filter( input, cv2.cvtColor(input.astype( 'uint8' ), cv2.COLOR_RGB2GRAY), kernel_s, sigma_r )

                while True:
                    while True:

                        # guidance
                        mask_r = (w[0] / sum( w )) * np.ones( input[:, :, 0].shape ).astype( 'float64' )
                        mask_g = (w[1] / sum( w )) * np.ones( input[:, :, 1].shape ).astype( 'float64' )
                        mask_b = (w[2] / sum( w )) * np.ones( input[:, :, 2].shape ).astype( 'float64' )
                        guidance = (np.multiply( mask_r, input[:, :, 0] ) + np.multiply( mask_g, input[:, :, 1] ) + np.multiply(mask_b, input[:, :, 2] ))
                        guidance = guidance.astype( 'uint8' )

                        # bf
                        jbf_out = self.bilateral_filter( input.astype( 'uint8' ), guidance, kernel_s, sigma_r)
                        set_r.append(w[0] / sum( w ))
                        set_g.append(w[1] / sum( w ))
                        set_b.append(w[2] / sum( w ))
                        dir_loss[str(set_r[n])+str(set_g[n])+str(set_b[n])] = np.sum( np.abs( jbf_out.astype( 'uint8' ) - jbf_gt.astype( 'uint8' ) ) )
                        # print(dir_loss)
                        n += 1
                        if w[1] == 0:
                            break
                        w[1] -= 1
                        w[2] += 1
                    if w[0] == 0 and w[1] == 0:
                        break
                    w[0] -= 1
                    w[1] += 10 - w[0]
                    w[2] = 0

                # local min
                local_min = []
                for i in range(len(set_r)):
                    value = dir_loss[str(set_r[i])+str(set_g[i])+str(set_b[i])]
                    local_check = True
                    if set_r[i] != 0.:  # down
                        if value > dir_loss[str(round(set_r[i]-0.1, 1))+str(round(set_g[i]+0.1, 1))+str(set_b[i])] or value > dir_loss[str(round(set_r[i]-0.1, 1))+str(set_g[i])+str(round(set_b[i]+0.1, 1))]:
                            local_check = False
                    if set_g[i] != 0.:  # mid1
                        if value > dir_loss[str(set_r[i])+str(round(set_g[i]-0.1, 1))+str(round(set_b[i]+0.1, 1))]:
                            local_check = False
                    if set_b[i] != 0.:  # mid2
                        if value > dir_loss[str(set_r[i])+str(round(set_g[i]+0.1, 1))+str(round(set_b[i]-0.1, 1))]:
                            local_check = False
                    if set_r[i] != 1.:  # up
                        if set_g[i] != 0.:  # mid1
                            if value > dir_loss[str( round(set_r[i]+0.1, 1) ) + str( round(set_g[i] - 0.1, 1) ) + str( set_b[i] )]:
                                local_check = False
                        if set_b[i] != 0.:  # mid2
                            if value > dir_loss[str( round(set_r[i]+0.1, 1) ) + str( set_g[i] ) + str( round(set_b[i] - 0.1, 1) )]:
                                local_check = False

                    if local_check:
                        local_min.append([set_r[i], set_g[i], set_b[i], dir_loss[str(set_r[i])+str(set_g[i])+str(set_b[i])]])
                print(local_min)





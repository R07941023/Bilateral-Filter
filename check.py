import numpy as np
import cv2
import pdb

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

        self.radius = 3 * self.sigma_s
        self.window_size = 2 * self.radius + 1


    def joint_bilateral_filter(self, input, guidance):

        guidance = np.array(guidance)
        # pdb.set_trace()

        if self.border_type == 'reflect':
            padded_input = cv2.copyMakeBorder(input, self.radius, self.radius, self.radius, self.radius, cv2.BORDER_REFLECT)
            padded_guidance = cv2.copyMakeBorder(guidance, self.radius, self.radius, self.radius, self.radius, cv2.BORDER_REFLECT)

        if self.border_type == 'replicate':
            padded_input = cv2.copyMakeBorder(input, self.radius, self.radius, self.radius, self.radius, cv2.BORDER_REPLICATE)
            padded_guidance = cv2.copyMakeBorder(guidance, self.radius, self.radius, self.radius, self.radius, cv2.BORDER_REPLICATE)

        if len(guidance.shape) == 2:
            # pdb.set_trace()
            #guidance = np.repeat(guidance[:, :, np.newaxis], 3, axis = 2)
            guidance = np.expand_dims(guidance, axis=2)
            padded_guidance = np.expand_dims(padded_guidance, axis=2)

        output = np.zeros(input.shape)

        spatial_kernel = np.zeros([self.window_size, self.window_size])
        for i, ii in enumerate(np.arange((-1) * self.radius, self.radius + 1)):
            for j, jj in enumerate(np.arange((-1) * self.radius, self.radius + 1)):
                spatial_kernel[i][j] = np.exp((-1) * (ii**2 + jj**2) / (2 * self.sigma_s**2))
        print( spatial_kernel.shape )
        print(spatial_kernel[0])
        exit()
        # pdb.set_trace()

        for x in range(input.shape[0]):
            padded_x = x + self.radius
            for y in range(input.shape[1]):
                padded_y = y + self.radius

                W = 0
                i_filtered = 0

                #x_range = np.arange(padded_x - self.radius, padded_x + self.radius + 1)
                #y_range = np.arange(padded_y - self.radius, padded_y + self.radius + 1)

                # guidance_box = padded_guidance[x_range][:, y_range] / 255.0
                # pdb.set_trace()
                guidance_box = padded_guidance[x: x+2*self.radius + 1, y: y+2*self.radius + 1, :] / 255.0
                # input_box = padded_input[x_range][:, y_range]
                input_box = padded_input[x: x+2*self.radius + 1, y: y+2*self.radius + 1, :]
                #pdb.set_trace()

                guidance_box = np.array(guidance_box)
                input_box = np.array(input_box)
                # pdb.set_trace()
                '''
                range_kernel = np.exp(((guidance_box[:, :, 0] - padded_guidance[padded_x, padded_y, 0] / 255.0)**2 + \
                                (guidance_box[:, :, 1] - padded_guidance[padded_x, padded_y, 1] / 255.0)**2 + \
                                (guidance_box[:, :, 2] - padded_guidance[padded_x, padded_y, 2] / 255.0)**2) * \
                                (-1) / (2 * self.sigma_r**2))
                '''
                range_kernel = np.exp(np.sum(np.square(guidance_box-padded_guidance[padded_x, padded_y]/ 255.0), axis = 2) / (-2 * self.sigma_r**2))
                # range_kernel = np.exp([a - b for a, b in guidance_box[:,:], padded_guidance[padded_x, padded_y]] * 
                #               (-1) / (2 * self.sigma_r**2))
                # pdb.set_trace()

                weight = np.multiply(range_kernel, spatial_kernel)
                # pdb.set_trace()

                weighted_input = input_box * np.repeat(weight[:,:,np.newaxis], 3, axis = 2)
                summed_weighted_input = weighted_input.sum(axis = 0).sum(axis = 0)

                output[x][y] = summed_weighted_input / np.sum(weight)

        return output
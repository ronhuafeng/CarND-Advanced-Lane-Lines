import numpy as np
import cv2


# This pipeline should work correctly on different images.
# Found issues:
# 1. results after HLS space transformation may differ on images read by cv2 and matplotlib.pyplot


def pipe_distort2undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def pipeline(img, sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Filter values from other blogs:
    # https://medium.com/@tjosh.owoyemi/finding-lane-lines-with-colour-thresholds-beb542e0d839
    # https://towardsdatascience.com/finding-lane-lines-on-the-road-30cf016a1165
    yellow_hls_low = np.array([20, 120, 100])
    yellow_hls_high = np.array([40, 200, 255])

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    hls_yellow = np.zeros_like(h_channel)
    hls_yellow[(h_channel >= yellow_hls_low[0]) & (h_channel <= yellow_hls_high[0]) &
               (s_channel >= yellow_hls_low[2])] = 1  # use higher s channel to filter yellow hills

    hls_white = np.zeros_like(l_channel)
    hls_white[l_channel > 200] = 1

    hls_binary = np.zeros_like(l_channel)
    hls_binary[(hls_yellow == 1) | (hls_white == 1)] = 1

    # Stack each channel
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, np.zeros_like(sxbinary))) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(hls_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, hls_binary


def pipe_undistort2edges(img_undistort):
    color_binary, combined_binary = pipeline(img_undistort, sx_thresh=(20, 100))
    return combined_binary


def pipe_edges2warped(img_edges, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_edges, M, (img_edges.shape[1], img_edges.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


# return mask around (center, (level+0.5) * height) with size (width, height)
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2

    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))
    left_centroids = [(l_center, 0)]
    right_centroids = [(r_center, 0)]

    # Go through each layer looking for max pixel locations
    for level in range(1, int(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference
        # is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_argmax = np.argmax(conv_signal[l_min_index:l_max_index])

        if conv_signal[l_argmax + l_min_index] > 100:
            l_center = l_argmax + l_min_index - offset
            left_centroids.append((l_center, level))

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_argmax = np.argmax(conv_signal[r_min_index:r_max_index])

        if conv_signal[r_argmax + r_min_index] > 100:
            r_center = r_argmax + r_min_index - offset
            right_centroids.append((r_center, level))

        # Add what we found for that layer
        # print('l max', l_argmax + l_min_index, conv_signal[l_argmax + l_min_index],
        #       'r max', r_argmax + r_min_index, conv_signal[r_argmax + r_min_index])
        window_centroids.append((l_center, r_center))
    return window_centroids, left_centroids, right_centroids


def pipe_warped2data(warped, window_width=50, window_height=80, margin=100):
    # window_width = 50
    # window_height = 80  # Break image into 9 vertical layers since image height is 720
    # margin = 100  # How much to slide left and right for searching

    window_centroids, left_centroids, right_centroids = find_window_centroids(warped, window_width, window_height,
                                                                              margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((warped, warped, warped)) * 255  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    ################## polyfit ################

    r_points = np.array([])
    l_points = np.array([])
    h_left_points = np.array([])
    h_right_points = np.array([])

    # print('left')
    for (p, level) in left_centroids:
        # print(p, level)
        l_points = np.append(l_points, p)
        h_left_points = np.append(h_left_points, warped.shape[0] - (level + 0.5) * window_height)

    # print('right')
    for (p, level) in right_centroids:
        # print(p, level)
        r_points = np.append(r_points, p)
        h_right_points = np.append(h_right_points, warped.shape[0] - (level + 0.5) * window_height)

    # print(l_points, r_points, h_points)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(h_left_points, l_points, 2)
    right_fit = np.polyfit(h_right_points, r_points, 2)

    ploty = np.linspace(
        start=0,
        stop=warped.shape[0] - 1,
        num=warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return output, ploty, left_fitx, right_fitx, left_fit, right_fit


def pipe_warped2origin(warped, undist, ploty, left_fitx, right_fitx, src, dst):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)

    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))

    # Combine the result with the original image
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def region_of_uninterest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.ones_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
    else:
        ignore_mask_color = 0

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def smooth_line(last_fits: list, fit, thresh=50):
    if len(last_fits) < 3:
        last_fits.append(fit)
    else:
        avg = sum([v[2] for v in last_fits]) / len(last_fits)
        if abs(fit[2] - avg) < thresh:
            last_fits.append(fit)

    if len(last_fits) > 3:
        last_fits = last_fits[-3:]
    return last_fits


def compute_curverad_offset(left_fitx, right_fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature

    y_eval = np.min(ploty)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    offset = (left_fitx[-1] + right_fitx[-1]) // 2 - 600

    return left_curverad, right_curverad, offset*xm_per_pix


def draw_curverad_offset(image, left_curverad, right_curverad, offset):
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(image, 'left_curverad %f m' % (left_curverad),
                (100, 100), font, 1.5, (255, 255, 255), 5, True)
    cv2.putText(image, 'right_curverad %f m' % (right_curverad),
                (100, 200), font, 1.5, (255, 255, 255), 5, True)
    cv2.putText(image, 'offset to center %f m' % (offset),
                (100, 300), font, 1.5, (255, 255, 255), 5, True)
    return image
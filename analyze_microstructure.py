import cv2
from cv2 import imwrite
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import clear_border
import glob
import os
import xlsxwriter
from scipy import ndimage
from scipy import stats as st
import PIL
from collections import Counter

# File path for reading images
path = 'microstructure_images/input/*.TIF'

bound = input('Name for dataset: ')

# Save stack data to single excel file
excel_name = ('microstructure_images/output/' + bound + '.xlsx')
#excel_scatter = ('microstructure_images/output_ver2/L3_1400_Alpha_Beta_Lower_Side/microstructure_scatter_' + filename + '_' + bound + '.xlsx')

# Keeping track of which image is being processed
image_in_stack = 0

# Distance transform factor for watershed method and image scale
transform_factor = 0.25
#pixels to micron
p_m = 60.3  # 30.15 #60.3 # 19.72
pixels_to_micron = 1 / p_m

# this is in units of um/pixel

# Initialize global data lists
glob_name = []
glob_pixel_to_micron = []
glob_filt = []
glob_boundary_tf = []

glob_alpha_frac = []
glob_alpha_thick = []
glob_alpha_thick_estimation = []
glob_mode_estimation = []
glob_median_estimation = []

glob_harmonic_thickness = []
glob_harmonic_thickness_weightage = []
glob_mean_beyond_mode_thickness = []
glob_mean_beyond_mode_weightage = []

# Open all images in "alpha_lath" folder
for file in glob.glob(path):
  num_of_images = len(glob.glob(path))
  image_number = []
  image_number_list = [x for x in range(0, num_of_images)]
  name = [os.path.basename(x) for x in glob.glob(path)]
  name_str = str(name[image_in_stack])
  size = len(name_str)
  # Slice string to remove .tif from string to save excel files
  filename = name_str[:size - 4]
  glob_name.append(filename)

  # Open the first image
  img1 = cv2.imread(file)
  horiz_dimension = img1.shape[0]
  vert_dimension = img1.shape[1]

  # To visually keep track of stack progress
  print('Image #', image_in_stack + 1, 'of', num_of_images, ':')

  # Set the crop dimensions for all images
  square_dimension = min(horiz_dimension, vert_dimension) - 200
  left = 200  #100
  right = left + square_dimension
  top = 0
  bottom = square_dimension
  crop_img = img1[top:bottom, left:right]
  #cv2.imshow("cropped", crop_img)

  # Convert to grayscale before thresholding
  img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
  #cv2.imshow('original', img)

  # Apply gaussian blur
  g_blur = cv2.GaussianBlur(img, (3, 3), 0)
  filter = g_blur

  # Apply CLAHE histogram correction
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
  cl_img = clahe.apply(filter)
  # Threshold image using OTSU method
  ret, thresh1 = cv2.threshold(cl_img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  #cv2.imshow('ostu', thresh1)

  # Apply dilation and erosion through opening method
  ksize = 1
  iter = 9
  kernel = np.ones((ksize, ksize), np.uint8)
  opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=iter)
  #cv2.imshow('opening',opening)

  mask_list = []
  for i in range(0, 19):
    transform_factor = 0.02 * i
    # Identify sure background pixels
    sure_bg = cv2.dilate(opening, kernel, iterations=iter)
    # Apply distance transform to define grains based on image intensity
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    # Identify sure foreground pixels by thresholding from distance transform
    ret2, sure_fg = cv2.threshold(dist_transform,
                                  transform_factor * dist_transform.max(), 255,
                                  0)
    sure_fg = np.uint8(sure_fg)
    # Define pixels that overlap foreground and background
    unknown = cv2.subtract(sure_bg, sure_fg)
    # cv2.imshow('unknown regions', unknown)
    # Connect nearby components
    ret3, markers = cv2.connectedComponents(sure_fg)

    # Set markers to nonzero value
    markers = markers + 10
    # Turn marker pixels that were in the unknown region to black
    markers[unknown == 255] = 0

    # Apply watershed method to marker pixels
    markers = cv2.watershed(crop_img, markers)
    # Identify only the background label
    beta_mask = (markers != 10)
    mask_list.append(beta_mask)
    # plt.imshow(beta_mask, cmap='Greys')
    # plt.show()

  selection = 0

  f, axarr = plt.subplots(1, 2, figsize=(15, 15), dpi=80)
  p = axarr[1].imshow(mask_list[selection], cmap='gray')
  p = axarr[0].imshow(crop_img)
  p = axarr[0].axis('off')
  p = axarr[1].axis('off')
  f.tight_layout()

  axSlider1 = plt.axes([0.2, 0.2, 0.25, 0.05])
  slder1 = Slider(axSlider1,
                  'Slider',
                  valmin=0,
                  valmax=19,
                  valinit=0.1,
                  valfmt='%1.0f',
                  valstep=1)

  def update(val):
    current_image = slder1.val
    selection = current_image
    axarr[1].imshow(mask_list[selection], cmap='gray')

  slder1.on_changed(update)

  plt.show()

  tf = np.float32(input('boundary transform factor: '))

  glob_boundary_tf.append(tf)
  glob_pixel_to_micron.append(p_m)

  # Remake the images so variables are saved correctly underneath
  transform_factor = tf * 0.02
  # Identify sure background pixels
  sure_bgf = cv2.dilate(opening, kernel, iterations=iter)
  # Apply distance transform to define grains based on image intensity
  dist_transformf = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
  # Identify sure foreground pixels by thresholding from distance transform
  ret2, sure_fgf = cv2.threshold(dist_transformf,
                                 transform_factor * dist_transformf.max(), 255,
                                 0)
  sure_fgf = np.uint8(sure_fgf)
  # Define pixels that overlap foreground and background
  unknown = cv2.subtract(sure_bgf, sure_fgf)
  # cv2.imshow('unknown regions', unknown)
  # Connect nearby components
  ret3, markersf = cv2.connectedComponents(sure_fgf)

  # Set markers to nonzero value
  markersf = markersf + 10
  # Turn marker pixels that were in the unknown region to black
  markersf[unknown == 255] = 0

  # Apply watershed method to marker pixels
  markersf = cv2.watershed(crop_img, markersf)
  # Identify only the background label
  beta_maskf = (markersf != 10)
  #plt.imshow(beta_maskf, cmap='Greys')
  #plt.show()

  # Save image from segmentation
  f, axarr = plt.subplots(1, 2, figsize=(15, 15), dpi=80)
  axarr[1].imshow(beta_maskf, cmap='gray')
  axarr[0].imshow(crop_img)
  axarr[0].axis('off')
  axarr[1].axis('off')
  f.tight_layout()
  #plt.show()
  plt.savefig('microstructure_images/output/' + filename + bound + '.png')

  # added from old code
  # Overlay marker pixels onto original images
  #crop_img[markersf == -1] = [0, 255, 255]
  # Show labels in color
  #img2 = color.label2rgb(markersf, bg_label=-1)
  #cv2.imshow('colored grains', img2)
  #cv2.imshow('outlined grains', crop_img)

  # Line measurement calculations
  thirty_degree_length = horiz_dimension
  sixty_degree_length = vert_dimension
  divisor = 100  #lines created
  factorsquare = square_dimension / divisor
  factorthirty = thirty_degree_length / divisor
  factorsixty = sixty_degree_length / divisor
  '''
    # making the code cleaner to do everything over one for loop instead of individually - to be worked on later
    angle_list = [0,30,60,90,120,150]

    for j in range(0, len(angle_list)):
        line_img = imgline + str(angle_list[j])
        line_img = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
        line_img.fill(255)
        for i in range(0, divisor*2):
    '''

  # 0 degree angle image
  img0 = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
  img0.fill(255)
  for i in range(0, divisor * 2):
    startx = 0
    starty = math.floor(factorsquare * i)
    endx = square_dimension
    endy = math.floor(factorsquare * i)
    cv2.line(img0, (startx, starty), (endx, endy), (0, 0, 0), 1)

  # 30 degree angle image
  img30 = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
  img30.fill(255)
  for i in range(0, divisor * 2):
    startx = 0
    starty = math.floor(factorthirty * i)
    endx = math.floor(factorthirty * i / (math.tan(math.pi / 6)))
    endy = 0
    cv2.line(img30, (startx, starty), (endx, endy), (0, 0, 0), 1)

  # 60 degree angle image
  img60 = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
  img60.fill(255)
  for i in range(0, divisor * 2):
    startx = 0
    starty = math.floor(factorsixty * i)
    endx = math.floor(factorsixty * i / (math.tan(math.pi / 3)))
    endy = 0
    cv2.line(img60, (startx, starty), (endx, endy), (0, 0, 0), 1)

  # 90 degree angle image
  img90 = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
  img90.fill(255)
  for i in range(0, divisor * 2):
    startx = math.floor(factorsquare * i)
    starty = 0
    endx = math.floor(factorsquare * i)
    endy = square_dimension
    cv2.line(img90, (startx, starty), (endx, endy), (0, 0, 0), 1)

  # 120 degree angle image
  img120 = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
  img120.fill(255)
  for i in range(0, divisor * 2):
    startx = square_dimension - math.floor(factorsixty * i /
                                           (math.tan(math.pi / 3)))
    starty = 0
    endx = square_dimension
    endy = math.floor(factorsixty * i)
    cv2.line(img120, (startx, starty), (endx, endy), (0, 0, 0), 1)

  # 150 degree angle image
  img150 = np.zeros((square_dimension, square_dimension), dtype=np.uint8)
  img150.fill(255)
  for i in range(0, divisor * 2):
    startx = square_dimension - math.floor(factorthirty * i /
                                           (math.tan(math.pi / 6)))
    starty = 0
    endx = square_dimension
    endy = math.floor(factorthirty * i)
    cv2.line(img150, (startx, starty), (endx, endy), (0, 0, 0), 1)

  # Subtract background from lines to get grain overlay
  zero_measure = img0 - beta_maskf
  thirty_measure = img30 - beta_maskf
  sixty_measure = img60 - beta_maskf
  ninety_measure = img90 - beta_maskf
  onetwenty_measure = img120 - beta_maskf
  onefifty_measure = img150 - beta_maskf

  # Add a border to the images for visual clarity
  bordersize = 200
  row, col = zero_measure.shape[:2]
  bottom = zero_measure[row - 2:row, 0:col]
  mean = cv2.mean(bottom)[0]
  zero_measure_buffer = cv2.copyMakeBorder(zero_measure,
                                           top=bordersize,
                                           bottom=bordersize,
                                           left=bordersize,
                                           right=bordersize,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=[mean, mean, mean])
  thirty_measure_buffer = cv2.copyMakeBorder(thirty_measure,
                                             top=bordersize,
                                             bottom=bordersize,
                                             left=bordersize,
                                             right=bordersize,
                                             borderType=cv2.BORDER_CONSTANT,
                                             value=[mean, mean, mean])
  sixty_measure_buffer = cv2.copyMakeBorder(sixty_measure,
                                            top=bordersize,
                                            bottom=bordersize,
                                            left=bordersize,
                                            right=bordersize,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=[mean, mean, mean])
  ninety_measure_buffer = cv2.copyMakeBorder(ninety_measure,
                                             top=bordersize,
                                             bottom=bordersize,
                                             left=bordersize,
                                             right=bordersize,
                                             borderType=cv2.BORDER_CONSTANT,
                                             value=[mean, mean, mean])
  onetwenty_measure_buffer = cv2.copyMakeBorder(onetwenty_measure,
                                                top=bordersize,
                                                bottom=bordersize,
                                                left=bordersize,
                                                right=bordersize,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=[mean, mean, mean])
  onefifty_measure_buffer = cv2.copyMakeBorder(onefifty_measure,
                                               top=bordersize,
                                               bottom=bordersize,
                                               left=bordersize,
                                               right=bordersize,
                                               borderType=cv2.BORDER_CONSTANT,
                                               value=[mean, mean, mean])

  # Display all of the generated image masks from all orientations
  '''
    cv2.imshow('0 degree image mask', zero_measure)
    cv2.waitKey(0)
    cv2.imshow('30 degree image mask', thirty_measure)
    cv2.waitKey(0)
    '''
  #cv2.imshow('60 degree image mask', sixty_measure)
  #cv2.imwrite('microstructure_images/output/' + filename + '_60degreemask', sixty_measure)
  #cv2.waitKey(0)

  plt.figure(figsize=(6, 4))
  plt.imshow(sixty_measure_buffer, cmap='gray')
  plt.axis('off')
  plt.tight_layout()
  #plt.show()
  plt.savefig('microstructure_images/output/' + filename + '_60degreemask' +
              '.png')
  '''
    cv2.imshow('90 degree image mask', ninety_measure)
    cv2.waitKey(0)
    cv2.imshow('120 degree image mask', onetwenty_measure)
    cv2.waitKey(0)
    cv2.imshow('150 degree image mask', onefifty_measure)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

  # initialize line measurement lists
  maj_zero_list = []
  maj_thirty_list = []
  maj_sixty_list = []
  maj_ninety_list = []
  maj_onetwenty_list = []
  maj_onefifty_list = []

  s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
  # Measure 0 degree segment lengths
  ret4, img0_measure = cv2.threshold(zero_measure_buffer, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  mask0 = img0_measure == 0
  labeled_mask0, num_labels0 = ndimage.label(mask0, structure=s)
  line_lengths0 = measure.regionprops(labeled_mask0, zero_measure_buffer)
  for i in range(1, len(line_lengths0)):
    maj_zero = line_lengths0[i].major_axis_length * 0.87 * pixels_to_micron
    maj_zero_list.append(maj_zero)

  # Measure 30 degree segment lengths
  ret5, img30_measure = cv2.threshold(thirty_measure_buffer, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  mask30 = img30_measure == 0
  labeled_mask30, num_labels30 = ndimage.label(mask30, structure=s)
  line_lengths30 = measure.regionprops(labeled_mask30, thirty_measure_buffer)
  for i in range(1, len(line_lengths30)):
    maj_thirty = line_lengths30[i].major_axis_length * 0.87 * pixels_to_micron
    maj_thirty_list.append(maj_thirty)

  # Measure 60 degree segment lengths
  ret6, img60_measure = cv2.threshold(sixty_measure_buffer, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  mask60 = img60_measure == 0
  labeled_mask60, num_labels60 = ndimage.label(mask60, structure=s)
  line_lengths60 = measure.regionprops(labeled_mask60, sixty_measure_buffer)
  for i in range(1, len(line_lengths60)):
    maj_sixty = line_lengths60[i].major_axis_length * 0.87 * pixels_to_micron
    maj_sixty_list.append(maj_sixty)

  # Measure 90 degree segment lengths
  ret7, img90_measure = cv2.threshold(ninety_measure_buffer, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  mask90 = img90_measure == 0
  labeled_mask90, num_labels90 = ndimage.label(mask90, structure=s)
  line_lengths90 = measure.regionprops(labeled_mask90, ninety_measure_buffer)
  for i in range(1, len(line_lengths90)):
    maj_ninety = line_lengths90[i].major_axis_length * 0.87 * pixels_to_micron
    maj_ninety_list.append(maj_ninety)

  # Measure 120 degree segment lengths
  ret8, img120_measure = cv2.threshold(onetwenty_measure_buffer, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  mask120 = img120_measure == 0
  labeled_mask120, num_labels120 = ndimage.label(mask120, structure=s)
  line_lengths120 = measure.regionprops(labeled_mask120,
                                        onetwenty_measure_buffer)
  for i in range(1, len(line_lengths120)):
    maj_onetwenty = line_lengths120[
        i].major_axis_length * 0.87 * pixels_to_micron
    maj_onetwenty_list.append(maj_onetwenty)

  # Measure 150 degree segment lengths
  ret8, img150_measure = cv2.threshold(onefifty_measure_buffer, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  mask150 = img150_measure == 0
  labeled_mask150, num_labels150 = ndimage.label(mask150, structure=s)
  line_lengths150 = measure.regionprops(labeled_mask150,
                                        onefifty_measure_buffer)
  for i in range(1, len(line_lengths150)):
    maj_onefifty = line_lengths150[
        i].major_axis_length * 0.87 * pixels_to_micron
    maj_onefifty_list.append(maj_onefifty)

  # Calculate average inverse of all line lengths
  # Create new master list to store all inverse values
  '''
    inv_master = []
    for i in range(len(maj_zero_list)):
        if maj_zero_list[i] == 0:
            inv0 = pixels_to_micron
            inv_master.append(inv0)
        else:
            inv0 = 1/maj_zero_list[i]
            inv_master.append(inv0)
    for i in range(len(maj_thirty_list)):
        if maj_thirty_list[i] == 0:
            inv30 = pixels_to_micron
            inv_master.append(inv30)
        else:
            inv30 = 1/maj_thirty_list[i]
            inv_master.append(inv30)
    for i in range(len(maj_sixty_list)):
        if maj_sixty_list[i] == 0:
            inv60 = pixels_to_micron
            inv_master.append(inv60)
        else:
            inv60 = 1/maj_sixty_list[i]
            inv_master.append(inv60)
    for i in range(len(maj_ninety_list)):
        if maj_ninety_list[i] == 0:
            inv90 = pixels_to_micron
            inv_master.append(inv90)
        else:
            inv90 = 1/maj_ninety_list[i]
            inv_master.append(inv90)
    for i in range(len(maj_onetwenty_list)):
        if maj_onetwenty_list[i] == 0:
            inv120 = pixels_to_micron
            inv_master.append(inv120)
        else:
            inv120 = 1/maj_onetwenty_list[i]
            inv_master.append(inv120)
    for i in range(len(maj_onefifty_list)):
        if maj_onefifty_list[i] == 0:
            inv150 = pixels_to_micron
            inv_master.append(inv150)
        else:
            inv150 = 1/maj_onefifty_list[i]
            inv_master.append(inv150)
    '''

  # Test to try filtering out very small line measurements
  norm_master = []
  inv_master = []
  #small_line_filter = 1*pixels_to_micron
  small_line_filter = 0.3
  glob_filt.append(small_line_filter)

  for i in range(len(maj_zero_list)):
    if maj_zero_list[i] <= small_line_filter:
      pass
    else:
      inv0 = 1 / maj_zero_list[i]
      norm_master.append(maj_zero_list[i])
      inv_master.append(inv0)
  for i in range(len(maj_thirty_list)):
    if maj_thirty_list[i] <= small_line_filter:
      pass
    else:
      inv30 = 1 / maj_thirty_list[i]
      norm_master.append(maj_thirty_list[i])
      inv_master.append(inv30)
  for i in range(len(maj_sixty_list)):
    if maj_sixty_list[i] <= small_line_filter:
      pass
    else:
      inv60 = 1 / maj_sixty_list[i]
      norm_master.append(maj_sixty_list[i])
      inv_master.append(inv60)
  for i in range(len(maj_ninety_list)):
    if maj_ninety_list[i] <= small_line_filter:
      pass
    else:
      inv90 = 1 / maj_ninety_list[i]
      norm_master.append(maj_ninety_list[i])
      inv_master.append(inv90)
  for i in range(len(maj_onetwenty_list)):
    if maj_onetwenty_list[i] <= small_line_filter:
      pass
    else:
      inv120 = 1 / maj_onetwenty_list[i]
      norm_master.append(maj_onetwenty_list[i])
      inv_master.append(inv120)
  for i in range(len(maj_onefifty_list)):
    if maj_onefifty_list[i] <= small_line_filter:
      pass
    else:
      inv150 = 1 / maj_onefifty_list[i]
      norm_master.append(maj_onefifty_list[i])
      inv_master.append(inv150)

  # Checking to see how many lines are generated for each orientation
  total_lengths = len(maj_zero_list) + len(maj_thirty_list) + len(
      maj_sixty_list) + len(maj_ninety_list) + len(maj_onetwenty_list) + len(
          maj_onefifty_list)
  #print(inv_master)
  print('length of master inverse list: ' + str(len(inv_master)))
  print('total number of lines created: ' + str(total_lengths))

  inv_master_round = []
  for y in inv_master:
    nearest_multiple = round((0.05 * round(y / 0.05)), 2)
    inv_master_round.append(nearest_multiple)

  #inv_master_round = [round(num, 2) for num in inv_master]
  norm_master_round = [round(num, 2) for num in norm_master]

  mean_n = round(np.mean(norm_master_round), 2)
  median_n = np.median(norm_master_round)

  #from scipy import stats as st
  mode_n = st.mode(norm_master_round)
  mode_print = mode_n[0]
  mode_count = mode_n[1]
  mode_print = str(mode_print)[1:-1]  #removing bracket
  mode_count = str(mode_count)[1:-1]  #removing bracket

  print('Mean of Norm master:', mean_n)
  print('Mode of Norm master:', mode_print)
  print('Count of Mode of Norm master:', mode_count)
  print('Median of Norm master:', median_n)

  #print(inv_master_round)

  x_scatter = []
  y_scatter = []
  x_scatteri = []
  y_scatteri = []
  for a in range(0, len(inv_master_round)):
    if inv_master_round[a] not in x_scatter:
      x_scatter.append(inv_master_round[a])

  x_scatter.sort()
  #print(x_scatter)
  #print(len(x_scatter))
  count = 0
  for i in range(0, len(x_scatter)):
    for j in range(0, len(inv_master_round)):
      if inv_master_round[j] == x_scatter[i]:
        count = count + 1
    y_scatter.append(count)
    count = 0

  #print(y_scatter)
  #print('Total Count:', sum(y_scatter))
  #t_count = sum(y_scatter) #total count
  #print(len(y_scatter))

  #mode_from scatter
  max_y = max(y_scatter)
  id_max = y_scatter.index(max_y)
  mod_inv = x_scatter[id_max]
  mod_length = 1 / mod_inv

  print('mode of line segment inverse:', mod_inv)
  print('mode of line segment in micron:', mod_length)

  # Sayem's Method

  x_scatter_mode = []
  y_scatter_mode = []
  for a in range(0, len(x_scatter)):
    if x_scatter[a] <= mod_inv:
      x_scatter_mode.append(x_scatter[a])
  count = 0

  for i in range(0, len(x_scatter_mode)):
    for j in range(0, len(x_scatter_mode)):
      if x_scatter_mode[j] == x_scatter_mode[i]:
        count = y_scatter[j]
    y_scatter_mode.append(count)
    count = 0

  #print(x_scatter_mode)
  #print(y_scatter_mode)

  #mode_from scatter
  max_y_mode = max(y_scatter_mode)
  id_max_mode = y_scatter.index(max_y_mode)
  mod_inv_mode = x_scatter[id_max_mode]
  mod_length_mode = 1 / mod_inv_mode

  print('mode of line segment inverse up to mode:', mod_inv_mode)

  # Sum of all inverse lengths and calculating mean

  sum_mode = []
  for i in range(0, len(x_scatter_mode)):
    mult = x_scatter_mode[i] * y_scatter_mode[i]
    sum_mode.append(mult)

  print(sum_mode)
  inv_mean_mode = sum(sum_mode) / sum(y_scatter_mode)
  print('mode of line segment in micron:', inv_mean_mode)

  # Calculating lath thickness from paper
  thick_triangular = 1 / (1.5 * inv_mean_mode)
  thick_triangular = round(thick_triangular, 2)
  print('thickness in triangular distribution is: ' + str(thick_triangular))
  w_t = sum(y_scatter_mode) / sum(y_scatter)  # weightage of triangular section
  w_t = round(w_t, 2)
  print('Weightage of Triangular: ' + str(w_t))

  # thickness_estimation beyond triangular distribution
  x_scatter_below_mode = []
  y_scatter_below_mode = []
  for a in range(0, len(x_scatter)):
    if x_scatter[a] > mod_inv:
      x_scatter_below_mode.append(x_scatter[a])
  count = 0

  for i in range(0, len(x_scatter_below_mode)):
    for j in range(0, len(x_scatter)):
      if x_scatter[j] == x_scatter_below_mode[i]:
        count = y_scatter[j]
    y_scatter_below_mode.append(count)
    count = 0

  print(x_scatter_below_mode)
  print(y_scatter_below_mode)

  sum_below_mode = []
  for i in range(0, len(x_scatter_below_mode)):
    mult = x_scatter_below_mode[i] * y_scatter_below_mode[i]
    sum_below_mode.append(mult)

  print(sum_below_mode)
  inv_mean_below_mode = sum(sum_below_mode) / sum(y_scatter_below_mode)

  thick_rest = 1 / (1.0 * inv_mean_below_mode)
  thick_rest = round(thick_rest, 2)
  print('thickness estimation beyond triangular distribution is: ' +
        str(thick_rest))
  w_bt = sum(y_scatter_below_mode) / sum(
      y_scatter)  # weightage of beyond triangular section
  w_bt = round(w_bt, 2)
  print('Weightage of Beyond Triangular Section: ' + str(w_bt))

  alpha_lath_estimation = round((thick_triangular * w_t + thick_rest * w_bt),
                                2)
  print('Alpha Lath Thickness Estimation in Micron: ' +
        str(alpha_lath_estimation))

  # Adding the thickness estimation_Sayem's method
  glob_alpha_thick_estimation.append(alpha_lath_estimation)
  glob_mode_estimation.append(mode_print)
  glob_median_estimation.append(median_n)
  glob_harmonic_thickness.append(thick_triangular)
  glob_harmonic_thickness_weightage.append(w_t)
  glob_mean_beyond_mode_thickness.append(thick_rest)
  glob_mean_beyond_mode_weightage.append(w_bt)

  excel_scatter = ('microstructure_images/output/' + filename + '_' + bound +
                   '.xlsx')
  outWorkbook_scatter = xlsxwriter.Workbook(excel_scatter)
  worksheet1 = outWorkbook_scatter.add_worksheet()

  # Creating New Scatter from the data for each image
  for file in glob.glob(path):
    worksheet1.write('A1', 'Inverse Length (1/micron)')
    worksheet1.write('B1', 'Frequency (Count)')

  for item in range(len(y_scatter)):
    worksheet1.write(item + 1, 0, x_scatter[item])
    worksheet1.write(item + 1, 1, y_scatter[item])

  worksheet2 = outWorkbook_scatter.add_worksheet('Count up to Mode')

  for file in glob.glob(path):
    worksheet2.write('A1', 'Inverse Length up to mode (1/micron)')
    worksheet2.write('B1', 'Frequency (Count)')

  for item in range(len(y_scatter_mode)):
    worksheet2.write(item + 1, 0, x_scatter_mode[item])
    worksheet2.write(item + 1, 1, y_scatter_mode[item])

  outWorkbook_scatter.close()

  # Rick's Methods (previous method)

  for c in x_scatter:
    x_scatteri.append(float(c))
  for d in y_scatter:
    y_scatteri.append(float(d))

  # Sum of all inverse lengths and calculating mean
  inv_mean = 0
  inv_mean = sum(inv_master) / len(inv_master)
  print('inverse mean is: ' + str(inv_mean))

  # Showing histogram of line segment size distribution
  df = pd.DataFrame(norm_master)
  plt.figure(figsize=(12, 8))
  plt.xlim([0, 5])
  #plt.ylim([0, 50])
  plt.hist(df[0], bins=5000, color='grey', edgecolor='black')
  #plt.title(title, fontsize=18, fontweight='bold')
  #plt.text(10, 40, text, fontsize=14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('Line Segment Size Distribution (micron)',
             fontsize=18,
             fontweight='bold')
  plt.ylabel('Count', fontsize=18, fontweight='bold')
  plt.tight_layout()
  plt.savefig(filename)
  plt.savefig('microstructure_images/output/' + filename + bound +
              '_lineSegment' + '.png')
  #plt.show()
  # plt.savefig('microstructure_images/output_ver2/L4_950_Alpha_Beta_Lower_Side/' + filename + '_60degreemask' + '.png')

  df = pd.DataFrame(inv_master_round)
  plt.figure(figsize=(12, 8))
  plt.xlim([0, 5])
  #plt.ylim([0, 50])
  plt.hist(df[0], bins=5000, color='grey', edgecolor='black')
  #plt.title(title, fontsize=18, fontweight='bold')
  #plt.text(10, 40, text, fontsize=14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('Inverse Line Segment Size Distribution (1/micron)',
             fontsize=18,
             fontweight='bold')
  plt.ylabel('Count', fontsize=18, fontweight='bold')
  plt.tight_layout()
  plt.savefig(filename)
  plt.savefig('microstructure_images/output/' + filename + bound +
              '_InverseSegment' + '.png')
  #plt.show()

  plt.figure(figsize=(12, 8))
  plt.xlim([0, 5])
  #plt.ylim([0, 50])
  plt.scatter(x_scatter, y_scatter, color='grey', edgecolor='black')
  #plt.title(title, fontsize=18, fontweight='bold')
  #plt.text(10, 40, text, fontsize=14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('Inverse Line Segment Size Distribution (1/micron)',
             fontsize=18,
             fontweight='bold')
  plt.ylabel('Count', fontsize=18, fontweight='bold')
  plt.tight_layout()
  plt.savefig(filename)
  plt.savefig('microstructure_images/output/' + filename + bound + '_Scatter' +
              '.png')
  #plt.show()

  # Calculating lath thickness from paper (previous method)
  thick = 1 / (1.5 * inv_mean)
  print('thickness is: ' + str(thick))
  glob_alpha_thick.append(thick)

  # Move on to the next image
  image_in_stack += 1

  # Initializing alpha vol fraction measurement
  regionsf = measure.regionprops(markersf, intensity_image=crop_img)
  propList = ['Area']
  area_list = []

  # Extracting % alpha information
  for i in range(1, len(regionsf)):
    # Save important variables
    area = regionsf[i].area * (pixels_to_micron**2)
    area_list.append(area)

    # Sum of all areas of beta ribs
    total_area = 0
    for i in range(1, len(area_list)):
      total_area = area_list[i] + total_area
    percent_alpha = 100 - (total_area /
                           (pixels_to_micron**2)) / (right * bottom) * 100
  glob_alpha_frac.append(percent_alpha)

# print('Alpha Fraction: ', glob_alpha_frac, ' % ')
# print('area:', area)

# Terminate master lists
glob_name.append('end')
glob_pixel_to_micron.append('end')
glob_filt.append('end')
glob_boundary_tf.append('end')

#glob_alpha_frac.append('end')
glob_alpha_thick.append('end')
glob_alpha_thick_estimation.append('end')
glob_mode_estimation.append('end')
glob_median_estimation.append('end')

glob_harmonic_thickness.append('end')
glob_harmonic_thickness_weightage.append('end')
glob_mean_beyond_mode_thickness.append('end')
glob_mean_beyond_mode_weightage.append('end')

# Set up excel sheet
outWorkbook = xlsxwriter.Workbook(excel_name)
outSheet = outWorkbook.add_worksheet('Detailed Alpha lath data')
outSheet.write('A1', 'image name')
outSheet.write('B1', 'Pixels to microns Ratio')
outSheet.write('C1', 'Boundary Transformation Factor')
outSheet.write('D1', 'Small Filter (micron)')
outSheet.write('F1', 'old alpha lath thickness (micron)')
outSheet.write('G1', 't_a, alpha lath thickness estimation (micron)')
outSheet.write('I1', 'mode (micron)')
outSheet.write('J1', 'median (micron)')
outSheet.write('L1', 't_ht, Harmonic Mean Thickness (micron)')
outSheet.write('M1', 'w_th, Harmonic Thickness Weightage')
outSheet.write('N1', 't_bt, Below mode thickness (micron)')
outSheet.write('O1', 'w_bt, Below mode Thickness Weightage')
#outSheet.write('C1', 'alpha volume fraction')

# Write lists to excel sheets
for item in range(len(glob_name)):
  outSheet.write(item + 1, 0, glob_name[item])
  outSheet.write(item + 1, 1, glob_pixel_to_micron[item])
  outSheet.write(item + 1, 2, glob_boundary_tf[item])
  outSheet.write(item + 1, 3, glob_filt[item])
  outSheet.write(item + 1, 5, glob_alpha_thick[item])
  outSheet.write(item + 1, 6, glob_alpha_thick_estimation[item])
  outSheet.write(item + 1, 8, glob_mode_estimation[item])
  outSheet.write(item + 1, 9, glob_median_estimation[item])
  outSheet.write(item + 1, 11, glob_harmonic_thickness[item])
  outSheet.write(item + 1, 12, glob_harmonic_thickness_weightage[item])
  outSheet.write(item + 1, 13, glob_mean_beyond_mode_thickness[item])
  outSheet.write(item + 1, 14, glob_mean_beyond_mode_weightage[item])

outWorkbook.close()
cv2.waitKey(0)
cv2.destroyAllWindows()

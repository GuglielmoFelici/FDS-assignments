# 1
def gauss(sigma):
    """
    Gauss function taking as argument the standard deviation sigma
    The filter should be defined for all integer values x in the range [-3sigma,3sigma]
    The function should return the Gaussian values Gx computed at the indexes x
    """

    return Gx, x

# 2


def box(filter_size=3, show_verbose=False):
    """
    box function taking as argument the filter size.
    The filter should be defined for all integer values and centered at zero
    The function should return the Box values Bx computed at the indexes x
    """

    return Bx, x

# 3


def custom():
    """
    This function returns the shown kernel.
    """

    return Gx

# 4


def gaussdx(sigma):

    return Dx, x

# 5


def gaussfiltering(img, sigma):
    """
    Implement a 2D Gaussian filter, leveraging the previous gauss.
    Implement the filter from scratch or leverage the convolve or convolve2D methods (scipy.signal)
    Leverage the separability of Gaussian filtering
    Input: image, sigma (standard deviation)
    Output: smoothed image
    """

    return np.array(smooth_img)

# 6


def boxfiltering(img, filter_size):
    """
    Implement a 2D Box filter, leveraging the previous box.
    Leverage the separability of Box filtering
    Input: image, filter_size
    Output: smoothed image
    """

    return np.array(smooth_img)

# 7


def customfiltering(img):
    """
    Implement a 2D Custom filter
    Implement the filter from scratch 
    Input: image
    Output: smoothed image
    """

    return np.array(smooth_img)

# 8


def downscale(img, factor):
    '''
    This function should return the given image
    resized by the factor.
    Input:
      img: the image to resize
      factor: the factor you want use to downscale
    Output:
      resized: the resized image
    '''

    return resized_img

# 9


def GaussianPyramid(img, steps=3, factor=0.5, sigma=4):
    '''
    This function implements the Gaussian Pyramid and shows the results.
    Leverage the "downscale" function.

    Inputs:
      img: the image
      steps: number of steps to use in performing the Pyramid
      factor: the scaling factor to resize the image at each step
      sigma: the Gaussian filter parameter

    Output:
      --
    '''
    return

# 10


def normalized_hist(img_gray, num_bins=40, show_verbose=False):
    '''
    Compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
    assume that image intensity is between 0 and 255

    img_gray - input image in grayscale format
    num_bins - number of bins in the histogram

    '''

    return hists, bins

# 11


def rgb_hist(img_color_double, num_bins=5, show_verbose=False):
    '''
    Compute the *joint* histogram for each color channel in the image
    The histogram should be normalized so that sum of all values equals 1
    Assume that values in each channel vary between 0 and 255

    img_color - input color image
    num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3

    E.g. hists[0,9,5] contains the number of image_color pixels such that:
        - their R values fall in bin 0
        - their G values fall in bin 9
        - their B values fall in bin 5
    '''

    return hists

# 12


def rg_hist(img_color_double, num_bins=5, show_verbose=False):
    '''
    Compute the *joint* histogram for the R and G color channels in the image
    The histogram should be normalized so that sum of all values equals 1
    Assume that values in each channel vary between 0 and 255

    img_color - input color image
    num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2

    E.g. hists[0,9] contains the number of image_color pixels such that:
        - their R values fall in bin 0
        - their G values fall in bin 9
    '''

    return hists

# 13


def gaussderiv(img, sigma):

    return imgDx, imgDy

# 14


def dxdy_hist(img_gray, num_bins=5, show_verbose=False):
    '''
    Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
    Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
    The histogram should be normalized so that sum of all values equals 1

    img_gray - input gray value image
    num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2

    Note: you can use the function gaussderiv from the Filtering exercise
    '''

    return hists

# 15


def dist_intersect(x, y):
    '''
    Compute the intersection distance between histograms x and y
  Return 1 - hist_intersection, so smaller values correspond to more similar histograms
  Check that the distance range in [0,1]
    '''

    return 1 - hist_intersection

# 16


def dist_l2(x, y):
    '''
    Compute the L2 distance between x and y histograms
    Check that the distance range in [0,sqrt(2)]
    '''

    return l2_dist

# 17


def dist_chi2(x, y):
    '''
    Compute chi2 distance between x and y
    Check that the distance range in [0,Inf]
    Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0
    '''

    return chi2_dist

# 18


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    '''
    this function returns a list containing the histograms for
    the list of images given as input.
    '''

    # note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain
#       handles to distance and histogram functions, and to find out whether histogram function
#       expects grayvalue or color image

    return image_hist

# 19


def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):
    '''
    Function to find the best match for each image in the 
    query folder.
    Input:
      model_images: list of strings with the path of model images.
      query_images: list of strings with the path of query images.
      dist_type:    a string to represent the name of the distance you want to 
                    use. Should be one among "l2", "intersect", "chi2".
      hist_type:    a string to represent the name of the histogram you want to 
                    use. Should be one among "grayvalue", "rgb", "rg", "dxdy".

    Output:
      best_match:   list containing in each position the index of the retrieved
                    best matching image.
      D:            Matrix with |model_images| rows and |query_images| columns
                    containing the scores of each matching.
    '''

    return best_match, D

# 20


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    '''
    For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.

    Note: use the previously implemented function 'find_best_match'
    Note: use subplot command to show all the images in the same Python figure, one row per query image

    '''

    return

# 21


def compute_KM_histograms(image_list, num_bins=10):
    '''
    Function to compute histograms of a list of images.
    Leverage the "rgb_hist" function previously coded.
    '''

    return image_hists

# 22


def compute_KMeans(model_list, query_list, K):
    '''
    Function to compute the recolored images from both query and 
    model lists. You should:
      1. Apply the fitting metod to queries retrieving the ist of recolored
         queries and KMeans models.
      2. Apply the transforming method on all the images in the model folder
         for each KMeans you fitted in the previous step.
      3. Return the 2 lists of recolored images.
    '''
    query_recolored_imgs = []
    query_km_models = []
    model_recolored_imgs = []

    return query_recolored_imgs, model_recolored_imgs

# 23


def compute_recolored_hists(query_recolored, model_recolored):
    '''
    Function to compute the histograms of lists of images.
    Use the compute_KM_histograms function.
    '''

    return query_hists, model_hists

# 24


def compute_matching(q_hists, m_hists, dist_type):
    '''
    Function to compute the scores among several histograms.
    Similarly to find_best_match function you have previously defined,
    this function will return the best_match list.

    EXTRA: add the 'all' dist_type wich compute the scores for all 3 
    kind of distances, exploiting the strength of modulation!
    '''

    return best_match

# 25


def print_corrects_KM(match, tot_q_images, dist_type, K):
    '''
    Function to print the results.
    Input:
      match: the best_match results
      tot_images: len(model_images)
      dist_type: among 'l2', 'intersect' and 'chi2' (EXTRA: 'all')
      K: K-Means parameter

    last three args should be expicitly printed by this function.

    Example of Output:

    Settings: K=30, dist=chi2
    Results: Number of correct matches: 58/89 (65.17%)
    '''


# 26


def KM_matching(model_imgs, query_imgs, dist_type, K):
    '''
    Use all the functions you have just defined to print the results.
    '''

    return

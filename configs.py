import os


class Config:
    # network meta params
    sr = 2.0
    scale_factors = [[sr, sr]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
    base_change_sfs = []  # list of scales after which the input is changed to be the output (recommended for high sfs)
    max_iters = 3000
    min_iters = 500
    min_learning_rate = 9e-8  # this tells the algorithm when to stop (specify lower than the last learning-rate)
    width = 64
    depth = 8
    output_flip = True  # geometric self-ensemble (see paper)
    downscale_method = 'cubic'  # a string ('cubic', 'linear'...), has no meaning if kernel given
    upscale_method = 'cubic'  # this is the base interpolation from which we learn the residual (same options as above)
    downscale_gt_method = 'cubic'  # when ground-truth given and intermediate scales tested, we shrink gt to wanted size
    learn_residual = False  # when true, we only learn the residual from base interpolation
    init_variance = 0.0  # variance of weight initializations, typically smaller when residual learning is on
    back_projection_iters = [10]  # for each scale num of bp iterations (same length as scale_factors)
    random_crop = True
    crop_size = 128
    noise_std = 0.0  # adding noise to lr-sons. small for real images, bigger for noisy images and zero for ideal case
    init_net_for_each_sf = False  # for gradual sr- should we optimize from the last sf or initialize each time?
    cuda = True
    # Params concerning learning rate policy
    learning_rate = 0.001
    learning_rate_change_ratio = 0.5  # ratio between STD and slope of linear fit, under which lr is reduced
    learning_rate_policy_check_every = 10
    learning_rate_slope_range = 256

    # Data augmentation related params
    augment_leave_as_is_probability = 0.1
    augment_no_interpolate_probability = 0.45
    augment_min_scale = 0.5
    augment_scale_diff_sigma = 0.45
    augment_shear_sigma = 0.8
    augment_allow_rotation = True  # recommended false for non-symmetric kernels

    # params related to test and display
    run_test = True
    run_test_every = 50  # 50
    display_every = 50
    name = 'test'
    plot_losses = False
    result_path = os.path.dirname(__file__) + '/results'
    create_results_dir = True
    input_path = local_dir = os.path.dirname(__file__) + '/input'
    create_code_copy = True  # save a copy of the code in the results folder to easily match code changes to results
    display_test_results = True
    save_results = True

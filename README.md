# RAISR.jl

RAISR implementation in julia

This is the julia implementation of the paper published by Yaniv Romano, John Isidoro, and Peyman Milanfar, Fellow, IEEE in Octo 2016 titled RAISR: Rapid and Accurate Image Super Resolution


Requirements

Pkg.add("Images")
Pkg.add("Distributions")
Pkg.add("JLD") - To write your weights/filters to a file after training
Pkg.add("Krylov") - For CGLS solver

To train on your own dataset replace the train data folders with your own code and run the the create_low_resolution_images_from_ground_truth.jl to createlow resolution images from your ground truth images 

You can use the code in the create_interpolated_images to create cheap linear upscaling of your images

train_patch_5_grad_9 and train_patch_11_grad_9 - are the training files they are pretty much the same code except for changes in the patch sizes and gradient patch sizes. Both training files have code to learn 8 free samples from each patch.

test_patch_11_9_with_free_training and test_patch_5_9_with_free_training are the notebooks using the filters to test some sample images. They are both the same code once again except for the difference in the patch sizes.There are different .jld learned filters on the sample data provided.

checkout the parallel_everywhere_function - it has a function that can take a whole image and learn the Q,V required . Remember, all you have to do is to combine the sum of the results of parallel execution into one big matrix Q/V. So you can learn filters on images parallely. It takes about 15 seconds on 1 core on mac per image

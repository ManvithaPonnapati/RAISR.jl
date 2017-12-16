
Pkg.add("Images")

using Images
using FileIO

GROUND_TRUTH_FILE = "/Users/manvithaponnapati/RAISR/train_data/ground_truth/"
LOW_RES_FILE = "/Users/manvithaponnapati/RAISR/train_data/low_res/"
filenames = readdir(GROUND_TRUTH_FILE)
file_full_paths = []
file_names = []
for file in filenames
  if (contains("$file", ".jpg")) == true
        push!(file_names,String(file))
        push!(file_full_paths,String(GROUND_TRUTH_FILE*file))
  end
end

println("List of ground truth images")
println(file_full_paths)

#To look at the images
#[load(file) for file in file_full_paths]

#Downscale images to a low resolution to use for training
DOWNSCALE_FACTOR = 2
for (index,file) in enumerate(file_full_paths)
    load_file = load(file)
    w,h= size(load_file)
    low_res_img = imresize(load_file,(convert(Int64,floor(w/DOWNSCALE_FACTOR)),convert(Int64,floor(h/DOWNSCALE_FACTOR))))
    save(LOW_RES_FILE*file_names[index],low_res_img)
end

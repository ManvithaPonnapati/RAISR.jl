
using Images

GROUND_TRUTH_FOLDER = "/Users/manvithaponnapati/RAISR/train_data/ground_truth/"
LOW_RES_FOLDER = "/Users/manvithaponnapati/RAISR/train_data/low_res/"
INTERPOLATED_FOLDER = "/Users/manvithaponnapati/RAISR/train_data/interpolated/"

#High Resolution Image
high_res_im = load(GROUND_TRUTH_FOLDER*"113009.jpg")

w,h = size(high_res_im)
println("Image dimensions $w and $h")

scale_factor = 2
scaled_dow_im = load(LOW_RES_FOLDER*"113009.jpg")
low_res_im = load(INTERPOLATED_FOLDER*"113009.jpg")
scaled_dow_im

[low_res_im high_res_im]

Pkg.add("Distributions")

using Distributions
#Calculate image properties
#Taking 3x3 patches on the imahes 
Qangle = 24
Qstrength = 3
Qcoherence = 3
patchsize = 3
Q = zeros((Qangle, Qstrength, Qcoherence, scale_factor*scale_factor, patchsize*patchsize, patchsize*patchsize))
V = zeros((Qangle, Qstrength, Qcoherence, scale_factor*scale_factor, patchsize*patchsize))
mark = zeros((Qstrength*Qcoherence, Qangle, scale_factor*scale_factor))

start_index = Int(ceil(patchsize/2))
weights_matrix = rand(Normal(0, 2), patchsize*patchsize)
weights_matrix = Array(Diagonal(weights_matrix))
#the really really twosted conversions on JuliaImages
colorview_rgb = convert(Array{Float64},channelview(low_res_im))
color_ve = 0.21*colorview_rgb[1,:,:]+ 0.72*colorview_rgb[2,:,:]+0.07*colorview_rgb[3,:,:]
w,h = size(color_ve)
for row in start_index:Int(w)-start_index
    for col in start_index:Int(h)-start_index
        patch = color_ve[row-1:row+1,col-1:col+1]
        gx,gy = imgradients(patch)
        gx = reshape(gx,patchsize*patchsize,1)
        gy = reshape(gy,patchsize*patchsize,1)
        GT = transpose([gx gy])
        GTWG = GT*weights_matrix*transpose(GT)
        eigen_max = eigmax(GTWG)
        eigen_min = eigmin(GTWG)
        eigen_vector_max = eigvecs(GTWG)[:,1]
        eigen_vector_min = eigvecs(GTWG)[:,2]
        gradient_angle = atan2(eigen_vector_max[2],eigen_vector_max[1])
        if gradient_angle < 0
            gradient_angle = gradient_angle + pi
        end
        lamda = abs(sqrt(complex(eigen_max)))/Qstrength
        u =(sqrt(complex(eigen_max)) -  sqrt(complex(eigen_min)))/(sqrt(complex(eigen_max)) +  sqrt(complex(eigen_min)))/Qcoherence
        angle = floor(gradient_angle/pi*Qangle)
      
        
        if lamda < 0.0001
            strength = 1
        elseif lamda > 0.001
            strength = 3
        else
            strength = 2
        end
           
        u = abs(u)
        if u < 0.25
            coherence = 1
        elseif u > 0.5
            coherence = 3
        else
            coherence = 2
        end
        
        
        # Bound the output to the desired ranges
        if angle > 23
            angle = 23
        elseif angle <= 0
            angle = 1
        end
        angle = Int(angle)
        # Get pixel type
        pixeltype = ((row-start_index) % scale_factor) * scale_factor + ((col-start_index) % scale_factor)
        pixelHR = color_ve[row,col]
        # Compute A'A and A'b
        wp,hp = size(patch)
        patch_1 = reshape(patch,wp*hp,1)
        patch = reshape(patch,wp*hp)
        ATA = dot(transpose(patch_1),patch_1)
        ATb = patch*pixelHR
        Q[angle,strength,coherence,pixeltype+1,:,:] += ATA
        V[angle,strength,coherence,pixeltype+1,:] += ATb
        mark[coherence*3+strength, angle, pixeltype] += 1
    end
end


println("Calculating filterS")
# Conjugate Gradients Solver
function cgls(A, b)
   
    height, width = size(A)
    
    x = zeros((height))
    while(true)
        sumA = sum(A)
        
        if (sumA < 100)
            break
        end
        if (det(A) < 1)
            A = A + eye(height, width) * sumA * 0.000000005
        else
            x = inv(A)*b
            break
        end
    end
    println("Return $x")
    return x
end

h = zeros((Qangle, Qstrength, Qcoherence, scale_factor*scale_factor, patchsize*patchsize))
operationcount = 0
totaloperations = scale_factor * scale_factor * Qangle * Qstrength * Qcoherence
for pixeltype in range(1, scale_factor*scale_factor)
    for angle in range(1, Qangle)
        for strength in range(1, Qstrength)
            for coherence in range(1, Qcoherence)
                println("index $pixeltype,$angle, $strength, $coherence")
                operationcount += 1
                h[angle,strength,coherence,pixeltype,:] = cgls(Q[angle,strength,coherence,pixeltype,:,:], V[angle,strength,coherence,pixeltype,:])
            end
        end
    end
end

using JLD
save("filterslearned.jld", "filters", h)

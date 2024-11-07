import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
from normals import load_model, calculate_normals

def load_hdr_image(file_path):
    hdr_image = Image.open(file_path)
    hdr_image = np.array(hdr_image).astype(np.float32)
    return hdr_image

output_dir = './samples/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_image(image, filename):
    """ Save the image in the output directory """
    output_path = os.path.join(output_dir, filename)
    plt.imsave(output_path, image)
    print(f"Image saved: {output_path}")

def display_image_in_color(image, title='', cmap='jet',  save_as=None):
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    color_image = plt.colormaps[cmap](norm_image)
    color_image = color_image[..., :3]
    plt.figure(figsize=(10, 5))
    plt.imshow(color_image)
    plt.title(title)
    plt.axis('off')
    if save_as:
        save_image(image, save_as)
    # plt.show()

def display_normals(normals_tensor, title='Normal Map',  save_as=None):
    normals_np = normals_tensor.squeeze(0).permute(1, 2, 0).numpy()
    normals_normalized = (normals_np - np.min(normals_np)) / (np.max(normals_np) - np.min(normals_np))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(normals_normalized)
    plt.title(title)
    plt.axis('off')
    if save_as:
        save_image(normals_normalized, save_as)
    # plt.show()

def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image

def illumination_map(hdr_image):
    hdr_image_normalized = normalize_image(hdr_image)
    luminance_map = np.mean(hdr_image_normalized, axis=-1)

    return luminance_map

def angular_areas(num_samples):
    # Area of each direction (solid angle), assuming uniform sampling
    total_area = 4 * np.pi  # Total solid angle for the entire sphere
    angular_area_per_direction = total_area / (num_samples**2)
    return angular_area_per_direction


# def get_light_direction():
#     return np.array([1.0, 0.0, 0.0])  # A fixed light direction

def get_light_direction(num_samples):
    directions = []
    dtheta = np.pi / num_samples
    dphi = 2 * np.pi / num_samples
    
    for i in range(num_samples):
        theta = (i + 0.5) * dtheta
        for j in range(num_samples):
            phi = (j + 0.5) * dphi
            Lx = np.sin(theta) * np.cos(phi)
            Ly = np.sin(theta) * np.sin(phi)
            Lz = np.cos(theta)
            directions.append(np.array([Lx, Ly, Lz]))
    
    return np.array(directions)

# Diffuse reflection is the scattering of light in many directions from a rough surface
def diffuse_reflection(I_map, N, A, kd):
    L_directions = get_light_direction(20)  
    diffuse_sum = np.zeros_like(I_map)
    # Lambert's Law: intensity of light reflected off a surface is directly proportional to the cosine of the angle between the light source and the surface normal
    for L in L_directions:
        N_dot_L = np.einsum('ijk,i->jk', N, L)
        fd = kd * np.maximum(N_dot_L, 0)
        diffuse_sum += I_map * fd * A

    D = diffuse_sum / (4 * np.pi)
    
    return D


# Specular reflection is the mirror-like reflection of light from a smooth surface, depends on the viewing angle
def specular_reflection(I, N, A, ks, n):
    S = np.zeros_like(I_map)
    L_directions = get_light_direction(20)
    N_transposed = np.transpose(N, (1, 2, 0))
    # Conventional Phong specular model
    for L in L_directions:
        N_dot_L = np.einsum('ijk,k->ij', N_transposed, L)
        R = 2 * N_transposed * N_dot_L[..., np.newaxis] - L
        R_dot_L = np.einsum('ijk,k->ij', R, L)
        fs = np.where(R_dot_L > 0, ks * (R_dot_L ** n), 0)
        S += I_map * fs * A
    
    S = S / (4 * np.pi)
    
    return S

# Output intensity combines both diffuse and specular reflections to give a realistic rendering of the scene
def output_intensity(D, S, N, Wd, Ws):
    E = np.array([0, 0, 1]) # desired viewing/eye direction
    N_transposed = np.transpose(N, (1, 2, 0))
    E_dot_N = np.einsum('ijk,k->ij', N_transposed, E)
    R = 2 * N_transposed * E_dot_N[:, :, np.newaxis] - E
    
    intensity = Wd * E_dot_N * D + Ws * E_dot_N * S
    
    intensity_min = intensity.min()
    intensity_max = intensity.max()
    intensity_normalized = (intensity - intensity_min) / (intensity_max - intensity_min)
    
    return intensity_normalized


print(os.getcwd())
file_path = os.path.abspath('samples/img/test1.hdr')
print(file_path)
hdr_image = load_hdr_image(file_path)
hdr_tensor = np.array(hdr_image)
hdr_tensor = normalize_image(hdr_tensor)

R = hdr_tensor[:, :, 0]
G = hdr_tensor[:, :, 1]
B = hdr_tensor[:, :, 2]  

I_map = illumination_map(hdr_image)
print("Illuminatin map done")
A = angular_areas(20)
print("Areas done")

model, device = load_model()
N = calculate_normals(file_path, model, device)
display_normals(N, title="Normals", save_as="Normals.png")
N = N.squeeze(0)
if len(N.shape) == 4:
    N = N[0]
print("Surface Normals done")

kd = 0.8
D = diffuse_reflection(I_map, N, A, kd)
print("Diffuse reflection map done.")
display_image_in_color(D, title="Diffuse Reflection Map in Color", cmap='jet', save_as="DR.png")

ks = 0.5
n = 10
S = specular_reflection(I_map, N, A, ks, n)
print("Specular reflection map done.")
display_image_in_color(S, title="Specular Reflection Map in Color", cmap='jet', save_as="SR.png")

Wd = 0.7
Ws = 0.3
intensity = output_intensity(D, S, N, Wd, Ws)
display_image_in_color(intensity, title="Final Intensity Map", cmap='jet', save_as="OI.png")


# to run : python lightmaps.py ./experiments/exp001_cvpr2024/dsine.txt

# Normals
# Red (R): Corresponds to the X-axis direction.
#     Red = 1 (Positive X): The normal is pointing right.
#     Red = 0 (Negative X): The normal is pointing left.
# Green (G): Corresponds to the Y-axis direction.
#     Green = 1 (Positive Y): The normal is pointing up.
#     Green = 0 (Negative Y): The normal is pointing down.
# Blue (B): Corresponds to the Z-axis direction (depth).
#     Blue = 1 (Positive Z): The normal is pointing towards the viewer (out of the screen).
#     Blue = 0 (Negative Z): The normal is pointing away from the viewer (into the screen).

# Reflection:
#     Bright areas (Yellow, Orange, Red): Represent regions where diffuse reflection is high.
#     Darker areas (Blue, Green): Represent regions where diffuse reflection is lower.

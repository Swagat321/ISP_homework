import os
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Read the image
image = io.imread('data/baby.tiff')

# Check and report how many bits per pixel the image has
bits_per_pixel = image.dtype.itemsize * 8
print(f"Bits per pixel: {bits_per_pixel}")

# Report its width and height
height, width = image.shape[:2]
print(f"Width: {width}, Height: {height}")

# Convert the image into a double-precision array
image_double = image.astype(np.float64)

# Convert the image into a linear array within the range [0, 1]
black_level = 0
white_level = 16383
r_scale, g_scale, b_scale = 1.628906, 1.000000, 1.386719

# Apply linear transformation to map the pixel values to the range [0, 1]
image_normalized = (image_double - black_level) / (white_level - black_level)

# Clip the values to the range [0, 1]
image_clipped = np.clip(image_normalized, 0, 1)

def white_world_wb(image):
    """Implement the white world white balancing algorithm."""
    max_value = np.max(image)
    wb_image = image.copy()
    wb_image /= max_value
    return np.clip(wb_image, 0, 1)

def gray_world_wb(image):
    """Implement the gray world white balancing algorithm."""
    mean = np.mean(image)
    wb_image = image.copy()
    wb_image /= mean
    return np.clip(wb_image, 0, 1)

def camera_preset_wb(image, r_scale, g_scale, b_scale):
    """Implement the camera's preset white balancing."""
    wb_image = image.copy()
    wb_image *= r_scale
    wb_image *= g_scale
    wb_image *= b_scale
    return np.clip(wb_image, 0, 1)

white_world_image = white_world_wb(image_clipped)
gray_world_image = gray_world_wb(image_clipped)
camera_preset_image = camera_preset_wb(image_clipped, r_scale, g_scale, b_scale)

# Create a figure with 1 row and 3 columns to display the images
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Display the original image
axes[0].imshow(image_clipped, cmap='gray')
axes[0].set_title('Original Image')

# Display the white world white balanced image
axes[1].imshow(white_world_image, cmap='gray')
axes[1].set_title('White World WB')

# Display the gray world white balanced image
axes[2].imshow(gray_world_image, cmap='gray')
axes[2].set_title('Gray World WB')

# Display the camera preset white balanced image
axes[3].imshow(camera_preset_image, cmap='gray')
axes[3].set_title('Camera Preset WB')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Display the figure
plt.show()

# Examine the top-left 2x2 region of the white-balanced images
top_left_2x2_white_world = white_world_image[:2, :2]
top_left_2x2_gray_world = gray_world_image[:2, :2]
top_left_2x2_camera_preset = camera_preset_image[:2, :2]

# Define the expected color patterns for the four Bayer patterns
bayer_patterns = {
    'grbg': np.array([[1, 0], [0, 1]]),
    'rggb': np.array([[0, 1], [1, 0]]),
    'bggr': np.array([[1, 0], [0, 1]]),
    'gbrg': np.array([[0, 1], [1, 0]]),
}

# Compare the 2x2 regions to the expected Bayer patterns
def compare_bayer_patterns(region):
    min_diff = float('inf')
    best_match = None
    for pattern, expected in bayer_patterns.items():
        diff = np.sum(np.abs(region - expected))
        if diff < min_diff:
            min_diff = diff
            best_match = pattern
    return best_match

white_world_match = compare_bayer_patterns(top_left_2x2_white_world)
gray_world_match = compare_bayer_patterns(top_left_2x2_gray_world)
camera_preset_match = compare_bayer_patterns(top_left_2x2_camera_preset)

print(f"White World Bayer pattern: {white_world_match}")
print(f"Gray World Bayer pattern: {gray_world_match}")
print(f"Camera Preset Bayer pattern: {camera_preset_match}")

def demosaic_rggb(image):
    """Demosaic the image using bilinear interpolation for the 'rggb' Bayer pattern."""
    height, width = image.shape
    
    # Create empty R, G, B channels
    r = np.zeros((height, width))
    g = np.zeros((height, width))
    b = np.zeros((height, width))
    
    # Fill in the R, G, B channels using bilinear interpolation
    r[::2, ::2] = image[::2, ::2]  # Red pixels
    g[::2, 1::2] = image[::2, 1::2]  # Green pixels in even rows
    g[1::2, ::2] = image[1::2, ::2]  # Green pixels in odd rows
    b[1::2, 1::2] = image[1::2, 1::2]  # Blue pixels
    
    # Use SciPy's interp2d to interpolate the missing values
    x = np.arange(width)
    y = np.arange(height)
    r_interp = interp2d(x, y, r, kind='linear')
    g_interp = interp2d(x, y, g, kind='linear')
    b_interp = interp2d(x, y, b, kind='linear')
    
    # Combine the R, G, B channels into a single RGB image
    rgb = np.dstack((r_interp(x, y), g_interp(x, y), b_interp(x, y)))
    
    return rgb

# Color Space Correction
def camera_to_srgb(camera_rgb):
    """Convert camera RGB values to linear sRGB."""
    # sRGB to XYZ matrix (provided in the sRGB standard)
    M_srgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])

    # Camera-specific matrix (from dcraw)
    M_xyz_to_cam = np.array([1.411621, -0.483647, 0.072026,
                             -0.105558, 1.005390, 0.100168,
                             -0.016250, 0.049537, 0.966713]) / 10000
    M_xyz_to_cam = M_xyz_to_cam.reshape(3, 3, order='C')

    # Compute the transformation matrix
    M_srgb_to_cam = np.dot(M_xyz_to_cam, M_srgb_to_xyz)

    # Normalize the transformation matrix
    M_srgb_to_cam /= M_srgb_to_cam.sum(axis=1, keepdims=True)

    # Apply the inverse transformation to convert from camera RGB to linear sRGB
    M_cam_to_srgb = np.linalg.inv(M_srgb_to_cam)
    srgb = np.dot(camera_rgb, M_cam_to_srgb.T)

    return srgb

# Demosaic the gray world white-balanced image using the 'rggb' Bayer pattern
demosaiced_gray_world = demosaic_rggb(gray_world_image)

# Convert the demosaiced image to sRGB color space
srgb_gray_world = camera_to_srgb(demosaiced_gray_world)

# Display the demosaiced image in sRGB color space
plt.imshow(srgb_gray_world)
plt.title('Gray World (Demosaiced & sRGB)')
plt.show()

#Brightness adjustment
def adjust_brightness(image, target_mean):
    """Adjust the brightness of the image to match the target mean intensity."""
    grayscale = color.rgb2gray(image)
    current_mean = np.mean(grayscale)
    scale_factor = target_mean / current_mean
    brightened_image = np.clip(image * scale_factor, 0, 1)
    return brightened_image

def gamma_encode(image):
    """Apply gamma encoding to the image."""
    encoded_image = np.where(image <= 0.0031308,
                             12.92 * image,
                             (1 + 0.055) * image ** (1/2.4) - 0.055)
    return encoded_image

# Brightness adjustment
target_mean = 0.25  # Adjust this value to your preference
brightened_gray_world = adjust_brightness(srgb_gray_world, target_mean)

# Gamma encoding
gamma_encoded_gray_world = gamma_encode(brightened_gray_world)

# Display the final image
plt.imshow(gamma_encoded_gray_world)
plt.title('Final Image (Gray World)')
plt.show()

# Convert gamma_encoded_image to uint8 to be able to save
gamma_encoded_gray_world = (gamma_encoded_gray_world * 255).astype(np.uint8)

# Save the image in PNG and JPEG formats
io.imsave('gray_world_final.png', gamma_encoded_gray_world)
io.imsave('gray_world_final_95.jpg', gamma_encoded_gray_world, quality=95)

# Calculate the compression ratio for quality=95
png_size_gray_world = os.path.getsize('gray_world_final.png')
jpg_95_size_gray_world = os.path.getsize('gray_world_final_95.jpg')
compression_ratio_95_gray_world = png_size_gray_world / jpg_95_size_gray_world
print(f"Compression ratio (quality=95) for Gray World: {compression_ratio_95_gray_world:.2f}")

# Find the lowest JPEG quality setting that is visually indistinguishable
qualities = list(range(95, 0, -5))
for quality in qualities:
    io.imsave(f'gray_world_final_{quality}.jpg', gamma_encoded_gray_world, quality=quality)
    
    jpg_size_gray_world = os.path.getsize(f'gray_world_final_{quality}.jpg')
    
    compression_ratio_gray_world = png_size_gray_world / jpg_size_gray_world
    
    print(f"Compression ratio (quality={quality}) for Gray World: {compression_ratio_gray_world:.2f}")
    
    if compression_ratio_gray_world > 10:
        break

def manual_white_balance(image, patch_coords):
    """Perform manual white balancing based on a selected patch."""
    patch_values = image[patch_coords[1], patch_coords[0]]
    r, g, b = patch_values
    
    # Handle zero values to avoid division by zero
    if r == 0:
        r = 1e-8
    if g == 0:
        g = 1e-8
    if b == 0:
        b = 1e-8
    
    # Calculate normalization factors
    r_norm = 1 / r
    g_norm = 1 / g
    b_norm = 1 / b
    
    # Normalize the color channels
    r_channel = image[:, :, 0] * r_norm
    g_channel = image[:, :, 1] * g_norm
    b_channel = image[:, :, 2] * b_norm
    
    # Clip the values to the range [0, 1]
    r_channel = np.clip(r_channel, 0, 1)
    g_channel = np.clip(g_channel, 0, 1)
    b_channel = np.clip(b_channel, 0, 1)
    
    # Combine the normalized channels into a new image
    balanced_image = np.dstack((r_channel, g_channel, b_channel))
    
    return balanced_image

# Display the demosaiced image and select a white patch
plt.imshow(demosaiced_gray_world)
plt.title('Select a white patch by clicking on the image')
patch_coords = plt.ginput(1)[0]
patch_coords = np.round(patch_coords).astype(int)
plt.close()

# Perform manual white balancing
balanced_image = manual_white_balance(demosaiced_gray_world, patch_coords)

# Display the original and white-balanced images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(demosaiced_gray_world)
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(balanced_image)
ax2.set_title('White Balanced Image')
ax2.axis('off')
plt.tight_layout()
plt.show()
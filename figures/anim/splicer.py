from PIL import Image
import os
import glob

# Folder containing the PNGs
image_folder = 'figures/anim'  # change this to your actual folder

# Pattern to match files like test_00001.png, test_00002.png, ...
image_pattern = os.path.join(image_folder, 'test_*.png')

# Get all matching filenames, sorted properly
image_files = sorted(glob.glob(image_pattern))

# Load images
frames = [Image.open(img) for img in image_files]

# Save as animated GIF
output_path = os.path.join(image_folder, 'preliminary_pseudoshock.gif')
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=250, loop=1)

print(f"GIF saved to {output_path}")
import os
from PIL import Image, ImageEnhance
# Input and output folder paths
input_folder = r"C:\Apollo\New_Data\18570 R14 88T AMZ 4G\dataset\Good\Train\patches_rtor"     # folder containing original images
output_folder = r"C:\Apollo\New_Data\18570 R14 88T AMZ 4G\dataset\Good\Train\patches_rtor\Augmentation"  # folder to save augmented images
# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
# Loop through each image in the input folder
def augment(input_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path)
        # Get base name without extension
            base_name = os.path.splitext(filename)[0]
        # Flip horizontally
            horizontal_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            horizontal_flip.save(os.path.join(output_folder, f"{base_name}_fH.jpg"))
        # Flip vertically
            vertical_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
            vertical_flip.save(os.path.join(output_folder, f"{base_name}_fpV.jpg"))
        # Increase brightness
            enhancer = ImageEnhance.Brightness(image)
            brighter = enhancer.enhance(1.5)
            brighter.save(os.path.join(output_folder, f"{base_name}_bt.png"))
        # Decrease brightness
            darker = enhancer.enhance(0.7)
            darker.save(os.path.join(output_folder, f"{base_name}_d.png"))
        print(f"Augmented: {filename}")
    print("\n🎉 All augmented images saved in:", output_folder)
import cv2
import os
import glob
from tqdm import tqdm

class RoadPreprocessor:
    def __init__(self, clip_limit=3.0, tile_size=(8, 8)):
        """
        Initializes the preprocessor with CLAHE parameters.
        :param clip_limit: Threshold for contrast limiting.
        :param tile_size: Size of the grid for histogram equalization.
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def apply_clahe(self, img):
        """
        Applies CLAHE to an image in the LAB color space to preserve natural colors.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L-channel (Lightness)
        cl = self.clahe.apply(l)
        
        # Merge channels back and convert to BGR
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_img

    def process_directory(self, input_dir, output_dir):
        """
        Processes all images in a directory and saves them to the output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_exts = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_exts:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"🖼️ Enhancing {len(image_files)} images...")
        for img_path in tqdm(image_files):
            img = cv2.imread(img_path)
            if img is not None:
                enhanced_img = self.apply_clahe(img)
                filename = os.path.basename(img_path)
                cv2.imwrite(os.path.join(output_dir, filename), enhanced_img)

if __name__ == "__main__":
    # Example usage for Milestone 6 logic
    processor = RoadPreprocessor()
    # Path relative to project root
    processor.process_directory('data/raw', 'data/processed')
    print("✅ Preprocessing complete. Enhanced images are in 'data/processed'.")

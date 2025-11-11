import os
import cv2
import numpy as np
from pathlib import Path


def find_corresponding_images(base_dir):
    """Find corresponding images across all directories with same filenames"""
    org_dir = Path(base_dir) / "org"
    vim_t_dir = Path(base_dir) / "vim_t"
    lbvim_t_dir = Path(base_dir) / "lbvim_t"
    lbvim_300_dir = Path(base_dir) / "lbvim_300"

    # Get all org images
    org_images = list(org_dir.glob("*.jpg"))

    corresponding_sets = []

    for org_img in org_images:
        # Get the exact filename
        filename = org_img.name

        # Look for corresponding images with same filename in other directories
        vim_t_img = vim_t_dir / filename
        lbvim_t_img = lbvim_t_dir / filename
        lbvim_300_img = lbvim_300_dir / filename

        # Check if all corresponding images exist
        if all(img.exists() for img in [vim_t_img, lbvim_t_img, lbvim_300_img]):
            corresponding_sets.append({
                'org': org_img,
                'vim_t': vim_t_img,
                'lbvim_t': lbvim_t_img,
                'lbvim_300': lbvim_300_img,
                'base_name': org_img.stem  # filename without extension
            })
        else:
            missing_files = []
            if not vim_t_img.exists():
                missing_files.append(f"vim_t/{filename}")
            if not lbvim_t_img.exists():
                missing_files.append(f"lbvim_t/{filename}")
            if not lbvim_300_img.exists():
                missing_files.append(f"lbvim_300/{filename}")

            print(f"Warning: Missing files for {filename}: {', '.join(missing_files)}")

    return corresponding_sets


def split_side_by_side_image(image):
    """Split a side-by-side image into left (gt) and right parts"""
    height, width = image.shape[:2]
    mid_point = width // 2

    left_part = image[:, :mid_point]  # Ground truth
    right_part = image[:, mid_point:]  # Actual result

    return left_part, right_part


def add_spacing(images, spacing=40):
    """Add spacing between images and concatenate horizontally"""
    if not images:
        return None

    # Get the maximum height to ensure all images have the same height
    max_height = max(img.shape[0] for img in images)

    # Resize all images to have the same height (keeping aspect ratio)
    resized_images = []
    for img in images:
        if img.shape[0] != max_height:
            # Calculate new width maintaining aspect ratio
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(max_height * aspect_ratio)
            img = cv2.resize(img, (new_width, max_height))
        resized_images.append(img)

    # Create spacing images (white background)
    if len(resized_images) > 1:
        spacing_img = np.ones((max_height, spacing, 3), dtype=np.uint8) * 255

        # Insert spacing between images
        final_images = [resized_images[0]]
        for img in resized_images[1:]:
            final_images.append(spacing_img)
            final_images.append(img)
    else:
        final_images = resized_images

    # Concatenate horizontally
    result = np.hstack(final_images)
    return result


def save_individual_images(images_dict, base_name, sep_dir):
    """Save individual images to the sep directory"""
    suffixes = ['org', 'gt', 'vim_t', 'lbvim_t', 'lbvim_300']

    for suffix, image in zip(suffixes, images_dict.values()):
        if image is not None:
            filename = f"{base_name}_{suffix}.jpg"
            output_path = sep_dir / filename

            success = cv2.imwrite(str(output_path), image)
            if success:
                print(f"  Saved individual: {filename}")
            else:
                print(f"  Failed to save individual: {filename}")


def process_images(base_dir, spacing=40):
    """Main function to process and concatenate images"""
    base_path = Path(base_dir)
    concat_dir = base_path / "concat"
    sep_dir = concat_dir / "sep"

    # Create directories
    concat_dir.mkdir(exist_ok=True)
    sep_dir.mkdir(exist_ok=True)

    # Find corresponding images
    corresponding_sets = find_corresponding_images(base_dir)

    if not corresponding_sets:
        print("No corresponding image sets found!")
        return

    print(f"Found {len(corresponding_sets)} corresponding image sets")

    for img_set in corresponding_sets:
        try:
            # Load org image
            org_img = cv2.imread(str(img_set['org']))
            if org_img is None:
                print(f"Failed to load org image: {img_set['org']}")
                continue

            # Load side-by-side images
            vim_t_img = cv2.imread(str(img_set['vim_t']))
            lbvim_t_img = cv2.imread(str(img_set['lbvim_t']))
            lbvim_300_img = cv2.imread(str(img_set['lbvim_300']))

            if any(img is None for img in [vim_t_img, lbvim_t_img, lbvim_300_img]):
                print(f"Failed to load some images for {img_set['base_name']}")
                continue

            # Split side-by-side images into gt and results
            gt_img, vim_t_result = split_side_by_side_image(vim_t_img)
            _, lbvim_t_result = split_side_by_side_image(lbvim_t_img)
            _, lbvim_300_result = split_side_by_side_image(lbvim_300_img)

            # Prepare images for concatenation
            images_to_concat = [
                org_img,
                gt_img,
                vim_t_result,
                lbvim_t_result,
                lbvim_300_result
            ]

            # Save individual images to sep directory
            individual_images = {
                'org': org_img,
                'gt': gt_img,
                'vim_t': vim_t_result,
                'lbvim_t': lbvim_t_result,
                'lbvim_300': lbvim_300_result
            }

            save_individual_images(individual_images, img_set['base_name'], sep_dir)

            # Add spacing and concatenate
            final_image = add_spacing(images_to_concat, spacing)

            # Save the concatenated result
            output_filename = f"{img_set['base_name']}_concat.jpg"
            output_path = concat_dir / output_filename

            success = cv2.imwrite(str(output_path), final_image)
            if success:
                print(f"Saved concatenated: {output_filename}")
            else:
                print(f"Failed to save concatenated: {output_filename}")

        except Exception as e:
            print(f"Error processing {img_set['base_name']}: {str(e)}")


def main():
    # Set your base directory path here
    base_directory = "/data07/shared/jzhang/result/mamba_bi/vim_IN1k_para/vis_det"

    # Process images with 40px spacing
    process_images(base_directory, spacing=40)

    print("Processing completed!")


if __name__ == "__main__":
    main()
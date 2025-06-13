import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_closing, binary_fill_holes, generate_binary_structure

def segment_femur_tibia(input_path, output_path, lower_hu=390, upper_hu=5000):
    """
    Segment knee CT scan into tibia (label 2), femur (label 1), and background (label 0).
    
    Args:
        input_path (str): Path to input CT scan (.nii.gz file).
        output_path (str): Path to save segmented mask (.nii.gz file).
        lower_hu (int): Lower Hounsfield Unit threshold for bone (default: 390).
        upper_hu (int): Upper Hounsfield Unit threshold for bone (default: 5000).
    """
    # Load CT data
    ct_img = nib.load(input_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine

    # Step 1: Thresholding to isolate bones
    bone_mask = np.logical_and(ct_data >= lower_hu, ct_data <= upper_hu)

    # Step 2: 2D slice-wise closing to avoid merging with fibula
    closed_mask = np.zeros_like(bone_mask)
    struct_2d = generate_binary_structure(2, 1)
    for z in range(bone_mask.shape[2]):
        closed_slice = binary_closing(bone_mask[:, :, z], structure=struct_2d, iterations=2)
        closed_mask[:, :, z] = closed_slice

    # Step 3: Label 3D components and select femur/tibia
    labeled_mask, _ = label(closed_mask)
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # Ignore background
    top_two_labels = np.argsort(component_sizes)[-2:]  # Select largest components

    # Step 4: Refine femur and tibia masks
    struct_3d = generate_binary_structure(3, 2)
    final_mask = np.zeros_like(closed_mask, dtype=np.uint8)
    for i, target_label in enumerate(top_two_labels, start=1):
        bone_mask = (labeled_mask == target_label)
        closed_bone = binary_closing(bone_mask, structure=struct_3d, iterations=3)
        filled_bone = binary_fill_holes(closed_bone)
        labeled_bone, num_features = label(filled_bone)
        if num_features > 1:
            largest_component = np.argmax(np.bincount(labeled_bone.ravel())[1:]) + 1
            filled_bone = (labeled_bone == largest_component)
        final_mask[filled_bone] = i  # Assign labels: 1=Femur, 2=Tibia

    # Save result
    output_img = nib.Nifti1Image(final_mask, affine)
    nib.save(output_img, output_path)
    print(f"Segmentation saved to {output_path}")
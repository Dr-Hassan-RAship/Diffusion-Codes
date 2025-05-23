# ------------------------------------------------------------------------------#
#
# File name                 : uncertainty_analysis.py
# Purpose                   : Analyze uncertainties in mask predictions (CCA and smoothness)
# Usage                     : Called after inference_ldm.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : May 23, 2025
# ------------------------------------------------------------------------------#
import os, csv
import numpy                    as np
import matplotlib.pyplot        as plt

from scipy.ndimage              import label, binary_opening
from skimage.measure            import perimeter, regionprops
from config                     import *
from utils                      import *
from inference_ldm_utils        import *

# ------------------------------------------------------------------------------#
def count_connected_components(mask):
    """Count the number of connected components in a binary mask after noise removal."""
    cleaned_mask = binary_opening(mask, structure=np.ones((3, 3)))
    labeled_array, num_features = label(cleaned_mask)
    return num_features

# ------------------------------------------------------------------------------#
def calculate_smoothness(mask, gt_mask=None):
    """Quantify boundary smoothness per component and overall, with Pred relative to GT if provided."""
    # Apply minimal noise removal to preserve distinct components
    cleaned_mask = binary_opening(mask, structure=np.ones((3, 3)))
    if cleaned_mask.sum() == 0:
        return ([np.nan], np.nan) if gt_mask is None else ([np.nan] * count_connected_components(gt_mask), np.nan)

    # Label the mask to identify components
    labeled_mask, num_features = label(cleaned_mask)
    props = regionprops(labeled_mask)
    if not props:
        return ([np.nan], np.nan) if gt_mask is None else ([np.nan] * count_connected_components(gt_mask), np.nan)

    smoothness_values = []
    for prop in props:
        area = prop.area
        component_mask = (labeled_mask == prop.label)
        perimeter_val = perimeter(component_mask.astype(np.uint8))
        smoothness = perimeter_val / (2 * np.sqrt(np.pi * area)) if area > 0 else np.nan
        if not np.isnan(smoothness):
            smoothness_values.append(smoothness)

    # Ensure the length matches the number of components detected by label
    expected_components = num_features if gt_mask is None else count_connected_components(gt_mask)
    while len(smoothness_values) < expected_components:
        smoothness_values.append(np.nan)

    overall_smoothness = np.mean(smoothness_values) if smoothness_values and not all(np.isnan(smoothness_values)) else np.nan

    if gt_mask is None:  # GT case
        return (smoothness_values, overall_smoothness)
    else:  # Pred case, relative to GT
        gt_components = count_connected_components(gt_mask)
        relative_smoothness = []
        for i in range(gt_components):
            if i < len(smoothness_values):
                relative_smoothness.append(smoothness_values[i] if not np.isnan(smoothness_values[i]) else np.nan)
            else:
                relative_smoothness.append(float('inf'))
        overall_smoothness = np.mean([s for s in relative_smoothness if not np.isinf(s)]) if any(not np.isinf(s) for s in relative_smoothness) else np.nan
        return (relative_smoothness, overall_smoothness)

# ------------------------------------------------------------------------------#
def analyze_uncertainty(folder_path):
    """Analyze CCA and smoothness for each patient subfolder."""
    results = []
    patient_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    for patient_id in patient_folders:
        patient_dir     = os.path.join(folder_path, patient_id)
        gt_mask_path    = os.path.join(patient_dir, "Mask_groundtruth.jpg")
        pred_mask_path  = os.path.join(patient_dir, "Mask_predicted.jpg")

        if not (os.path.exists(gt_mask_path) and os.path.exists(pred_mask_path)):
            continue

        # Load and process masks (assuming binary, convert to numpy array)
        gt_mask = plt.imread(gt_mask_path)[:, :, 0] > 0.5  # Threshold to binary
        pred_mask = plt.imread(pred_mask_path)[:, :, 0] > 0.5  # Threshold to binary

        # CCA analysis
        gt_components   = count_connected_components(gt_mask)
        pred_components = count_connected_components(pred_mask)
        component_diff  = abs(gt_components - pred_components)

        # Smoothness analysis
        gt_smoothness_tuple = calculate_smoothness(gt_mask)
        pred_smoothness_tuple = calculate_smoothness(pred_mask, gt_mask)

        # Convert tuples to strings for CSV
        gt_smoothness_str   = str(gt_smoothness_tuple)
        pred_smoothness_str = str(pred_smoothness_tuple)
        smoothness_diff    = abs(gt_smoothness_tuple[1] - pred_smoothness_tuple[1]) if not np.isnan(gt_smoothness_tuple[1]) and not np.isnan(pred_smoothness_tuple[1]) else np.nan

        results.append([patient_id, gt_components, pred_components, component_diff, gt_smoothness_str, pred_smoothness_str, smoothness_diff])
        
    # Save results to CSV
    output_csv = os.path.join(folder_path, "uncertainty_metrics.csv")
    headers = ["Patient_ID", "GT_Components", "Pred_Components", "Component_Diff", "GT_Smoothness", "Pred_Smoothness", "Smoothness_Diff"]
    check_or_create_folder(folder_path)
    
    save_metrics_to_csv(results, output_csv, headers = headers)
    
    print(f"✅ Uncertainty metrics saved to {output_csv}")

# ------------------------------------------------------------------------------#
if __name__ == "__main__":
#     # Provide the inference output folder path
    inference_folder = os.path.join(do.SAVE_FOLDER, f"inference_test_M{do.MODEL_EPOCHS[0]}")
    # inference_folder = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/Diffusion-Codes/Diffusers-Testing/testing_analysis'
    analyze_uncertainty(inference_folder)
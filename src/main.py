import os
import nibabel as nib
from scipy.ndimage import zoom
import torchvision.models as models
import pickle
import pandas as pd
from segmentation import segment_femur_tibia
from model_inflation import inflate_densenet2d_to_3d
from feature_extraction import FeatureExtractor
from cosine_similarity import compute_cosine_similarity

def run_pipeline(ct_path, seg_path, output_features_path="features.pkl", output_csv_path="similarity_results.csv"):
    """
    Run the full pipeline: segmentation, feature extraction, comparison, and result saving.
    
    Args:
        ct_path (str): Path to input CT scan (.nii.gz file).
        seg_path (str): Path to segmentation mask (.nii.gz file).
        output_features_path (str): Path to save extracted features (default: 'features.pkl').
        output_csv_path (str): Path to save cosine similarity results (default: 'similarity_results.csv').
    """
    # Step 1: Segment the CT scan if not already done
    if not os.path.exists(seg_path):
        segment_femur_tibia(ct_path, seg_path)

    # Load and downsample volumes
    ct_volume = nib.load(ct_path).get_fdata()
    seg_volume = nib.load(seg_path).get_fdata()
    ct_volume = zoom(ct_volume, zoom=(0.5, 0.5, 0.5), order=1)
    seg_volume = zoom(seg_volume, zoom=(0.5, 0.5, 0.5), order=0)

    # Step 2: Load and inflate DenseNet121 model
    model_2d = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
    model_3d = inflate_densenet2d_to_3d(model_2d, time_dim=3)

    # Step 3: Extract features
    extractor = FeatureExtractor(model_3d)
    features = extractor.extract_features(ct_volume, seg_volume)
    extractor.cleanup()

    # Save extracted features
    with open(output_features_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Features extracted and saved to {output_features_path}")

    # Step 4: Compute cosine similarities
    region_pairs = [('tibia', 'femur'), ('tibia', 'background'), ('femur', 'background')]
    layers = ['last_conv', 'third_last_conv', 'fifth_last_conv']
    results_list = []

    for region1, region2 in region_pairs:
        row = {'image_name': os.path.basename(ct_path), 'region_pair': f"{region1}_{region2}"}
        if region1 not in features or region2 not in features:
            print(f"Warning: One or both regions ({region1}, {region2}) missing in features")
            for layer in layers:
                row[layer] = np.nan
        else:
            for layer in layers:
                if layer not in features[region1] or layer not in features[region2]:
                    print(f"Warning: Layer {layer} missing for {region1} or {region2}")
                    row[layer] = np.nan
                else:
                    vec1 = features[region1][layer]
                    vec2 = features[region2][layer]
                    similarity = compute_cosine_similarity(vec1, vec2)
                    row[layer] = similarity
                    print(f"Cosine similarity for {region1}_{region2}_{layer}: {similarity}")
        results_list.append(row)

    # Step 5: Save results to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Cosine similarities saved to {output_csv_path}")

if __name__ == "__main__":
    ct_path = "3702_left_knee.nii.gz"
    seg_path = "femur_tibia_segmentation.nii.gz"
    run_pipeline(ct_path, seg_path)
import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from deepdisc.data_format.conversions import convert_to_json

plt.style.use('seaborn-v0_8-whitegrid')

def combine_metadata(annotations_dir, all_metadata_path):
    """Loads and combines all dc2_*.json files from a directory into a single file"""
    print(f"Combining all annotation files from: {annotations_dir}")
    combined_data = []
    if not os.path.isdir(annotations_dir):
        print(f"Error: Annotation directory not found at {annotations_dir}")
        return None
        
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.json') and filename.startswith('dc2'):
            file_path = os.path.join(annotations_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                combined_data.extend(data)
    
    with open(all_metadata_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
        
    print(f"Saved {len(combined_data)} total entries to {all_metadata_path}")
    return all_metadata_path


def clean_metadata(all_metadata_path, clean_metadata_path):
    """Filters out invalid entries (e.g., empty images) from the combined metadata"""
    print(f"Cleaning metadata from: {all_metadata_path}")
    with open(all_metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    valid_cutouts = [entry for entry in all_metadata if entry['height'] > 0 and entry['width'] > 0]
    
    with open(clean_metadata_path, 'w') as f:
        json.dump(valid_cutouts, f, indent=4)
        
    print(f"Saved {len(valid_cutouts)} valid entries to {clean_metadata_path}")
    return clean_metadata_path


def split_and_save_data(clean_metadata_path, output_dir, train_ratio=0.7, val_ratio=0.1):
    """Shuffles, splits, and saves the data into train, validation, and test sets."""
    print("Splitting data into train, validation, and test sets...")
    
    with open(clean_metadata_path, 'r') as f:
        all_data = json.load(f)
    
    random.shuffle(all_data)
    total_size = len(all_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    print(f"  - Train size: {train_size} images")
    print(f"  - Validation size: {val_size} images")
    print(f"  - Test size: {total_size - train_size - val_size} images")
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    train_path = os.path.join(output_dir, 'train.json')
    val_path = os.path.join(output_dir, 'val.json')
    test_path = os.path.join(output_dir, 'test.json')

    print(f"\nSaving split datasets to: {output_dir}")
    convert_to_json(train_data, train_path)
    convert_to_json(val_data, val_path)
    convert_to_json(test_data, test_path)
    print("... Done.")


def generate_plots(output_dir, snr_lvl, band):
    """Generates and saves summary plots for the data splits."""
    print("Generating and saving summary plots...")

    train_path = os.path.join(output_dir, 'train.json')
    val_path = os.path.join(output_dir, 'val.json')
    test_path = os.path.join(output_dir, 'test.json')

    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # Histogram of objects per image
    train_objs = [len(d['annotations']) for d in train_data]
    val_objs = [len(d['annotations']) for d in val_data]
    test_objs = [len(d['annotations']) for d in test_data]

    plt.figure(figsize=(10, 6))
    plt.hist(train_objs, bins=10, color='blue', alpha=0.5, label=f'Training Set ({sum(train_objs)})')
    plt.hist(val_objs, bins=10, color='green', alpha=0.5, label=f'Validation Set ({sum(val_objs)})')
    plt.hist(test_objs, bins=10, color='orange', alpha=0.5, label=f'Test Set ({sum(test_objs)})')
    plt.xlabel('Number of Objs')
    plt.ylabel('Number of Images')
    plt.title(f'Histogram of Objects per Image - Lvl {snr_lvl}')
    plt.legend()
    plot1_path = os.path.join(output_dir, f'obj_count_hist_lvl{snr_lvl}.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f"  - Saved obj count histogram to {plot1_path}")

    # Magnitude distributions
    def extract_mag(data, band):
        mag = 'mag_'+band
        stars_mags, galaxies_mags, all_mags = [], [], []
        for anns in (d['annotations'] for d in data):
            for ann in anns:
                val = ann[mag]
                all_mags.append(val)
                (galaxies_mags if ann['category_id'] == 0 else stars_mags).append(val)
                
        return stars_mags, galaxies_mags, all_mags

    def plot_hist(ax, data, bins, color, label, title):
        """
        Plots a histogram on a given Axes object and adds vertical lines for mean and max magnitudes
        Args:
            ax (matplotlib.axes.Axes): The axes object to plot on
            data (list): data to plot
            bins (np.array): bins for histogram
            color (str): Color of histogram
            label (str): Label for data 
            title (str): Title for subplot
        """
        ax.hist(data, bins=bins, color=color, alpha=0.8, label=label)  
        if data:
            mean_val = np.nanmean(data)
            max_val = np.nanmax(data)
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(max_val, color='red', linestyle='--', linewidth=2, label=f'Max: {max_val:.2f}')
        ax.set_xlabel(r'mag $i$')
        ax.set_ylabel('Number of Objs')
        if title:
            ax.set_title(title)
        ax.legend(frameon=True)

    train_stars_i, train_galaxies_i, train_mags_i = extract_mag(train_data, band)
    val_stars_i, val_galaxies_i, val_mags_i = extract_mag(val_data, band)
    test_stars_i, test_galaxies_i, test_mags_i = extract_mag(test_data, band)
    
    all_stars_i = train_stars_i + val_stars_i + test_stars_i
    all_galaxies_i = train_galaxies_i + val_galaxies_i + test_galaxies_i
    all_mags_i = train_mags_i + val_mags_i + test_mags_i
    
    if not all_galaxies_i and not all_stars_i:
        print("Warning: No magnitude data found to plot.")
        return
        
    bins = np.linspace(min(all_mags_i), max(all_mags_i), 30)
    colors = {
        'train': '#3498db',  # blue
        'val': '#2ecc71',    # green
        'test': '#e67e22',   # orange
        'stars': '#e74c3c',  # red
        'galaxies': '#00008B',  # dark blue
        'all': '#34495e'     # Dark Gray
    }
    
    # train, val, test
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig1.suptitle(r'mag $i$ Distribution by Dataset' + f' - Lvl {snr_lvl}', fontsize=14)

    plot_data_1 = {
        'Training': (train_mags_i, colors['train']),
        'Validation': (val_mags_i, colors['val']),
        'Test': (test_mags_i, colors['test'])
    }

    for i, (split_name, (data, color)) in enumerate(plot_data_1.items()):
        plot_hist(axs1[i], data, bins=bins, color=color, label=f'{split_name} (n={len(data)})', title=f'{split_name} Set')

    fig1.tight_layout()
    
    fig1_path = os.path.join(output_dir, f'mag{band}_dist_split_lvl{snr_lvl}.png')
    fig1.savefig(fig1_path)
    plt.close(fig1)
    print(f"  - Saved mag i distribution by dataset to {fig1_path}")

    # All Objs
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    title = r'Overall Distribution of All Objects by mag $i$' + f' - Lvl {snr_lvl}'
    plot_hist(ax2, all_mags_i, bins=bins, color=colors['all'], label=f'All Objs (n={len(all_mags_i)})', title=title)
    
    fig2_path = os.path.join(output_dir, f'mag{band}_dist_lvl{snr_lvl}.png')
    fig2.savefig(fig2_path)
    plt.close(fig2)
    print(f"  - Saved mag i distribution of all annotations to {fig2_path}")
    
    # galaxies and stars + train, val, test
    fig3, axs3 = plt.subplots(3, 2, figsize=(18, 22), sharex=True)
    fig3.suptitle(r'mag $i$ Distribution by Source and Dataset' +  f'- Lvl {snr_lvl}', fontsize=16)

    plot_data_2 = {
        'Training': (train_galaxies_i, train_stars_i),
        'Validation': (val_galaxies_i, val_stars_i),
        'Test': (test_galaxies_i, test_stars_i)
    }
    
    for i, (split_name, (gal_data, star_data)) in enumerate(plot_data_2.items()):
        plot_hist(axs3[i, 0], gal_data, bins=bins, color=colors['galaxies'], label=f'{split_name} Galaxies (n={len(gal_data)})', title=f'{split_name}: Galaxies')
        plot_hist(axs3[i, 1], star_data, bins=bins, color=colors['stars'], label=f'{split_name} Stars (n={len(star_data)})', title=f'{split_name}: Stars')

    fig3.tight_layout()
    
    fig3_path = os.path.join(output_dir, f'mag{band}_dist_source_lvl{snr_lvl}.png')
    fig3.savefig(fig3_path)
    plt.close(fig3)
    print(f"  - Saved mag i distribution by source type to {fig3_path}")

def main():
    """Main function to run the data preprocessing pipeline"""
    parser = argparse.ArgumentParser(description="Prepare and split LSST annotation data for DeepDISC training.")
    parser.add_argument('--root_dir', type=str, default='./lsst_data/',
                        help='Root directory where the lsst_data is stored.')
    parser.add_argument('--snr_lvl', type=int, default=5,
                        help='SNR level of the annotations to process.')
    parser.add_argument('--plots', action='store_true',
                        help='If set, generate and save summary plots.')
    parser.add_argument('--band', type=str, default='i',
                        help='Band to use for the magnitude distribution plots.')
    
    args = parser.parse_args()

    annotations_dir = os.path.join(args.root_dir, f'annotations_lvl{args.snr_lvl}/')
    all_metadata_path = os.path.join(annotations_dir, 'all_metadata_incl_empty.json')
    clean_metadata_path = os.path.join(annotations_dir, 'all_metadata.json')
    
    if not os.path.exists(clean_metadata_path):
        combine_metadata(annotations_dir, all_metadata_path)
        clean_metadata(all_metadata_path, clean_metadata_path)
    else:
        print(f"Cleaned metadata file already exists: {clean_metadata_path}")

    # split data and save JSON files
    train_path = os.path.join(annotations_dir, 'train.json')
    if not os.path.exists(train_path):
        split_and_save_data(clean_metadata_path, annotations_dir)
    else:
        print(f"Train/Val/Test splits already exist in: {annotations_dir}")

    # generate plots if requested
    if args.plots:
        generate_plots(annotations_dir, args.snr_lvl, args.band)

if __name__ == "__main__":
    main()
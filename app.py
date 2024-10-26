# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Configure folder paths
IMAGE_FOLDER = '/home/jupyter/face_labelling/static/images'
ORIGINAL_PARQUET_FILE = '/home/jupyter/face_labelling/training_data_balanced.parquet'
NEW_PARQUET_FILE = '/home/jupyter/face_labelling/new_annotation.parquet'

# Global variables
df = None
new_annotations_df = None

def load_dataframes():
    global df, new_annotations_df
    
    # Load original parquet file
    try:
        df = pd.read_parquet(ORIGINAL_PARQUET_FILE)
        print(f"Successfully loaded original parquet file with {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading original parquet file: {str(e)}")
        raise

    # Load or create new annotations parquet file
    try:
        if os.path.exists(NEW_PARQUET_FILE):
            new_annotations_df = pd.read_parquet(NEW_PARQUET_FILE)
            print(f"Loaded existing new annotations file with {len(new_annotations_df)} records")
        else:
            new_annotations_df = pd.DataFrame(columns=['image_name', 'new_labels'])
            new_annotations_df.to_parquet(NEW_PARQUET_FILE, index=False)
            print("Created new annotations file")
    except Exception as e:
        print(f"Error with new annotations file: {str(e)}")
        raise

# Load dataframes at startup
load_dataframes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/get_image_list', methods=['GET'])
def get_image_list():
    try:
        image_list = df['image_name'].tolist()
        verified_images = []
        for img in image_list:
            full_path = os.path.join(IMAGE_FOLDER, img)
            if os.path.exists(full_path):
                verified_images.append(img)
        return jsonify(verified_images)
    except Exception as e:
        print(f"Error in get_image_list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_annotations', methods=['GET'])
def get_annotations():
    global new_annotations_df, df
    
    image_name = request.args.get('image_name')
    if not image_name:
        return jsonify({'error': 'No image name provided'}), 400

    try:
        # Check if this image has new annotations
        new_annotations = new_annotations_df[new_annotations_df['image_name'] == image_name]
        
        if len(new_annotations) > 0:
            # Image exists in new annotations
            print(f"Found new annotations for {image_name}")
            labels = new_annotations['new_labels'].iloc[0]
        else:
            # No new annotations, get from original dataset
            print(f"No new annotations found for {image_name}, using original")
            original_data = df[df['image_name'] == image_name]
            if len(original_data) == 0:
                return jsonify({'error': 'Image not found in any dataset'}), 404
            labels = original_data['label'].iloc[0]
        
        # Convert numpy arrays to lists for JSON serialization
        processed_labels = []
        for box in labels:
            if isinstance(box, np.ndarray):
                box_list = box.tolist()
            else:
                box_list = list(box)
            processed_labels.append([float(x) for x in box_list])

        return jsonify({
            'annotations': processed_labels,
            'source': 'new' if len(new_annotations) > 0 else 'original'
        })
    except Exception as e:
        print(f"Error getting annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    global new_annotations_df
    
    try:
        data = request.json
        if not data or 'image_name' not in data or 'annotations' not in data:
            return jsonify({'error': 'Invalid data provided'}), 400

        image_name = data['image_name']
        new_annotations = data['annotations']

        # Print debug information
        print(f"Saving annotations for {image_name}")
        print(f"Received annotations: {new_annotations}")

        # Convert to numpy arrays
        numpy_annotations = [np.array(box, dtype=np.float32) for box in new_annotations]
        
        # If image exists in new_annotations_df, delete it first
        new_annotations_df = new_annotations_df[new_annotations_df['image_name'] != image_name]
        
        # Create a new row and append it
        new_row = pd.DataFrame({
            'image_name': [image_name],
            'new_labels': [numpy_annotations]
        })
        
        # Append the new row
        new_annotations_df = pd.concat([new_annotations_df, new_row], ignore_index=True)
        
        # Save to parquet
        new_annotations_df.to_parquet(NEW_PARQUET_FILE, engine='pyarrow', index=False)
        
        print(f"Successfully saved annotations to {NEW_PARQUET_FILE}")
        print(f"Current records in new_annotations_df: {len(new_annotations_df)}")
        print(f"Saved annotation type: {type(numpy_annotations)}")
        
        # Verify the save
        saved_df = pd.read_parquet(NEW_PARQUET_FILE)
        saved_record = saved_df[saved_df['image_name'] == image_name]
        if len(saved_record) > 0:
            print(f"Verified save - found record for {image_name}")
        else:
            print("Warning: Save verification failed - record not found")
            
        return jsonify({'success': True})

    except Exception as e:
        print(f"Error saving annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Print sample records
    if len(df) > 0:
        sample_image = df['image_name'].iloc[0]
        print("\nSample record:")
        print(f"Image name: {sample_image}")
        print(f"Original labels type: {type(df['label'].iloc[0])}")
        print(f"Original labels shape: {df['label'].iloc[0].shape if hasattr(df['label'].iloc[0], 'shape') else 'N/A'}")
        
        # Check if we have new annotations for this image
        new_sample = new_annotations_df[new_annotations_df['image_name'] == sample_image]
        if not new_sample.empty:
            sample_labels = new_sample['new_labels'].iloc[0]
            print(f"New labels type: {type(sample_labels)}")
            print(f"New labels shape: {sample_labels.shape if hasattr(sample_labels, 'shape') else 'N/A'}")
        else:
            print("No new labels for this image yet")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
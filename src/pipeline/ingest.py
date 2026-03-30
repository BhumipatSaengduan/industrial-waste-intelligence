import json
import pandas as pd
import re
from pathlib import Path

DATA_PATH = 'data/raw'

def parse_date(filename):
    match = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace('-', ':')
        return pd.to_datetime(f'{date_str} {time_str}')
    return None

def load_annotations(splits=['train', 'valid', 'test']):
    """Load and flatten COCO annotations into a single dataframe."""
    all_records = []

    for split in splits:
        json_path = f'{DATA_PATH}/{split}/_annotations.coco.json'
        with open(json_path) as f:
            data = json.load(f)

        cat_map = {c['id']: c['name'] for c in data['categories']}
        img_map = {img['id']: img for img in data['images']}

        for ann in data['annotations']:
            img       = img_map[ann['image_id']]
            cls       = cat_map.get(ann['category_id'], 'unknown')

            # Exclude parent class and near-zero bbox
            if cls == 'wastes':
                continue
            bbox_area = ann['bbox'][2] * ann['bbox'][3]
            if bbox_area < 1.0:
                continue

            dt = parse_date(img['file_name'])

            all_records.append({
                'annotation_id' : ann['id'],
                'image_id'      : ann['image_id'],
                'filename'      : img['file_name'],
                'split'         : split,
                'class_name'    : cls,
                'bbox_x'        : round(ann['bbox'][0], 2),
                'bbox_y'        : round(ann['bbox'][1], 2),
                'bbox_width'    : round(ann['bbox'][2], 2),
                'bbox_height'   : round(ann['bbox'][3], 2),
                'bbox_area'     : round(bbox_area, 2),
                'image_width'   : img['width'],
                'image_height'  : img['height'],
                'datetime'      : dt,
                'date'          : dt.date() if dt else None,
                'hour'          : dt.hour if dt else None,
            })

    df = pd.DataFrame(all_records)
    print(f'Total records loaded: {len(df)}')
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'Classes: {df["class_name"].unique().tolist()}')
    return df

if __name__ == '__main__':
    df = load_annotations()

    # Save as CSV
    Path('data/processed').mkdir(exist_ok=True)
    out_path = 'data/processed/annotations_processed.csv'
    df.to_csv(out_path, index=False)
    print(f'Saved: {out_path} ({len(df)} rows)')

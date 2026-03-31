import pandas as pd
import yaml
import os

def load_params(config_path="params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def find_mri_path(mri_id, base_paths, relative_path):
    for base in base_paths:
        full_path = os.path.join(base, mri_id, relative_path)
        if os.path.exists(full_path):
            return full_path
    return None

def main():
    # Load configuration
    params = load_params()
    raw_cfg = params['raw_data']
    load_cfg = params['load_options']

    # 1. Load and clean DataFrame
    df = pd.read_excel(raw_cfg['excel_path'])
    df = df[load_cfg['columns']].dropna()

    # 2. Apply Labeling logic (HC = 0, AD = 1)
    df['label'] = df['CDR'].apply(lambda x: 0 if x == 0 else 1)

    # 3. Map MRI Paths
    df['path'] = df['MRI ID'].apply(
        lambda x: find_mri_path(x, raw_cfg['base_paths'], raw_cfg['img_relative_path'])
    )
    
    # Drop rows where MRI file was not found
    df = df.dropna(subset=['path'])

    # 4. Save the output for the next DVC stage
    os.makedirs(os.path.dirname(load_cfg['output_path']), exist_ok=True)
    df.to_csv(load_cfg['output_path'], index=False)
    
    print(f"Total samples: {len(df)}")
    print(f"Metadata saved to: {load_cfg['output_path']}")

if __name__ == "__main__":
    main()
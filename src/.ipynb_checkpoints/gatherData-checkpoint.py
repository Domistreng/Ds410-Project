import kagglehub
import shutil
import os

randomState = 38


## DOWNLOADING LARGE DATASET
print("Attempting to download dataset")
if not os.path.exists('../data/raw/US_Accidents_March23.csv'):

    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")


    for root, dirs, files in os.walk(path):
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join('../data/raw', file)
            shutil.move(src_file, '../data/raw')
            print(f"Moved: {file}")
            
    shutil.rmtree(path)
    
    print ("Downloaded raw data")
    
else:
    print ("Raw data already downloaded")


## BREAK OFF SAMPLE SET FOR LOCAL
print("Attempting to sample dataset")
if not os.path.exists('../data/processed/sample_df.csv'):
    sampleSize = 20000
    chunksize = 100000

    import pandas as pd
    
    
    sampled_chunks = []
    rows_sampled = 0
    for chunk in pd.read_csv('../data/raw/US_Accidents_March23.csv', chunksize=chunksize):
        rows_remaining = sampleSize - rows_sampled
        if rows_remaining <= 0:
            break

        frac = min(1.0, rows_remaining / len(chunk))
        chunk_sampled = chunk.sample(frac=frac, random_state=randomState)
        
        sampled_chunks.append(chunk_sampled)
        rows_sampled += len(chunk_sampled)
        
    sample_df = pd.concat(sampled_chunks, ignore_index=True)
    sample_df.to_csv('data/processed/sample_df.csv', index=False)
    
    print ("Data sampled for local testing")
    
else:
    print("Database sampled")
    

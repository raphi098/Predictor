import os

files = os.listdir('extracted_frames')

for file in files:
    os.remove('extracted_frames/' + file)
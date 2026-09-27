import os
import re

file_path = 'backend-ml/app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'CHECKPOINT_PATH = \([^)]+\)', 
    'CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "best_segformer_oil_spill.pth")', 
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

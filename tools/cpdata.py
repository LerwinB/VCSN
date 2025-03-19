import os
import shutil
srcpath = 'data/Realdataset/Soyuz/label'
tarpath = 'data/Realdataset/label'
files=os.listdir(srcpath)
for file in files:
    shutil.copy(os.path.join(srcpath,file),os.path.join(tarpath,'Soyuz'+file))
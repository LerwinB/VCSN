
txtfile='/home/buaa22/Software/zhaoshengyun/mmsegmentation/data/uesatL/uesatL_10_random.txt'
with open(txtfile,'r') as file:
    lines=file.readlines()

cleaned_names = [name.replace('.png', '') for name in lines]
with open(txtfile, 'w') as file:
    file.writelines(cleaned_names)
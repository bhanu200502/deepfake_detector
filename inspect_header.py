import os
with open('model.h5','rb') as f:
    header=f.read(16)
print('header bytes:', header)
print('hex:', [hex(b) for b in header])
print('size', os.path.getsize('model.h5'))

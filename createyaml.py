import os
yamlfile=open("dec.yaml","w",encoding='utf-8')
nc=os.listdir('new_train')
yamlfile.write("names:\n")

for i in nc:
    yamlfile.write("- "+i+'\n')
yamlfile.write(f"nc:{len(nc)}\n")
yamlfile.write('train: \n')
yamlfile.write('val: \n ')
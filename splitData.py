import os
import random
import shutil
from itertools import islice

outputFolderPath="Dataset/SplitData"
inputFolderPath="Dataset/all"
splitRatio={"train":0.7,"val":0.2,"test":0.1}
classes = ["fake","real"]
try:
    shutil.rmtree(outputFolderPath)

except OSError as r:
    os.mkdir(outputFolderPath)

#-------Directory to Create -----
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)
#-------Get the names-----
listNames=os.listdir(inputFolderPath)
print(listNames)
print(len(listNames))

uniqueNames=[]
for  name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames=list(set(uniqueNames))
#-------Shuffle -----
random.shuffle(uniqueNames)
#-------find the number od images for each Folder -----
lenData=len(uniqueNames)

lenTrain=int(lenData*splitRatio['train'])
lenVal=int(lenData*splitRatio['val'])
lenTest=int(lenData*splitRatio['test'])
#-------put the remaining images in Training -----
if lenData != lenTrain+lenTest+lenVal:
    remaining=lenData-(lenTrain+lenTest+lenVal)
    lenTrain+=remaining
#print(f'Total Images:{lenData} \n Split {lenTrain} {lenVal} {lenTest}')
#----Split the List---------
lengthToSplit = [lenTrain,lenVal,lenTest]
Input=iter(uniqueNames)
Output=[list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \n Split {len(Output[0])} {len(Output[1])} {len(Output[2])}')
#-------Copy thr files-----
sequence = ['train', 'val', 'test']
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')
print("Split Process Completed...")


# -------- Creating Data.yaml file  -----------

dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")

#%%
import numpy as np
import pandas as pd
from skimage.feature import blob_log
from matplotlib import pyplot as plt
from IPython.display import display

#%%
# Generate blob image
L=100
blob_N=500
sigma=3
xx, yy, zz = np.meshgrid(np.arange(L), np.arange(L), np.arange(L))

blob_image = np.zeros((L,L,L),dtype=np.float64)
blob_positions=np.random.rand(blob_N,3)*L

for blob in blob_positions:
    blob_image += np.exp(-((xx-blob[0])**2+(yy-blob[1])**2+(zz-blob[2])**2)/sigma**2)

blob_image+=np.random.normal(0,0.1,blob_image.shape)

plt.imshow(blob_image[:,:,L//2])
# %%
# Generate label image

label_image=np.zeros((L,L,L),dtype=np.uint8)
pos_radius=[((10,10,40),20),((50,50,50),30),((80,20,60),10)]
for j, (pos, radius) in enumerate(pos_radius):
    label_image[(xx-pos[0])**2+(yy-pos[1])**2+(zz-pos[2])**2<radius**2]=j+1
for i in range(L//2-20,L//2+20,10):
    plt.imshow(label_image[:,:,i],vmin=0,vmax=3)
    plt.show()
# %%
# Detect blobs

blob_positions_data=blob_log(blob_image, min_sigma=3, 
                            max_sigma=4, num_sigma=1, 
                            threshold=0.05)
blob_positions_df=pd.DataFrame(
                        blob_positions_data,
                        columns=['x','y','z','sigma'])


# %%
# Get label values for each blob

blob_positions_df["ix"]=blob_positions_df["x"].round()
blob_positions_df["iy"]=blob_positions_df["y"].round()
blob_positions_df["iz"]=blob_positions_df["z"].round()
blob_positions_df["label"]=label_image[
    blob_positions_df["ix"].astype(int),
    blob_positions_df["iy"].astype(int),
    blob_positions_df["iz"].astype(int)
]
blob_positions_df["label"]=blob_positions_df["label"].astype(int)
#for i, row in blob_positions_df.iterrows():
#    blob_positions_df.loc[i,"label"]=label_image[
#        int(row["ix"]),int(row["iy"]),int(row["iz"])]

for i in range(L//2-20,L//2+20,10):
    plt.imshow(label_image[:,:,i],vmin=0,vmax=3)
    df=blob_positions_df[
        (blob_positions_df["iz"]==i)
        & (blob_positions_df["label"]==2)
    ]
    plt.plot(df["x"],df["y"],'o')
    plt.show()
# %%
# count blobs for each label
label_counts=blob_positions_df["label"].value_counts()
label_counts_df=pd.DataFrame(label_counts).sort_index()
label_counts_df.index.name="label"
label_counts_df.columns=["count"]
display(
   label_counts_df
)
# %%

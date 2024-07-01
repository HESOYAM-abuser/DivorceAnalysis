import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = 'divorce-csv.csv'
data = pd.read_csv(path, sep=',')
df = data.drop(['Class'], axis=1)

res = PCA(n_components=3).fit_transform(df)
res = pd.DataFrame(res, columns=['PC1','PC2','PC3'])

fig = plt.figure().add_subplot(projection='3d')
fig.set_title("PCA on the dataset")
fig.set_xlabel("PC1")
fig.set_ylabel("PC2")
fig.set_zlabel("PC3")

df['GMM'] = GMM(2).fit(res).predict(res)
df['KMeans'] = KMeans(2).fit(res).predict(res)
df['Class'] = data['Class']

clr = [
    [0]*df.shape[0],
    df['GMM'],
    df['KMeans'],
    df['Class'],
][3]

r = [res[i] for i in res.columns]
fig.scatter(*r, c=clr, cmap="PRGn")
plt.show()

df.to_csv('output.csv', index=False)
#%matplotlib inline
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
#insurance = pd.read_csv(r'C:\Users\Nalin\Desktop\insurance.csv')
#print(insurance.columns)
#print(insurance.head(10))
#print(insurnce.describe())
tableau_20 = [(31,119,180), (174,199,232), (255,127,14), (255,187, 120),(44,160,44),(152,223,138),(214,39,40),(255,152,150),(148,103,189),(197,176,213),(140,86,75),(196,156,148),(227,119,194),(247,182,210),(127,127,127),(199,199,199),(188,189,34),(219,219,141),(23,190,207),(158,218,229)]
for i in range(len(tableau_20)):
    r,g,b = tableau_20[i]
    tableau_20[i]=(r/255.,g/255.,b/255.)
#print(tableau_20[4])
#sns.lmplot(x="age", y="charges", data=insurance, scatter_kws={'color':tableau_20[2], 'alpha':0.75})
#tips = sns.load_dataset("tips")
#sns.jointplot("total_bill", "tip", data=tips,kind="reg", color=tableau_20[16])
#sns.jointplot("total_bill", "tip", data=tips, kind="hex", color=tableau_20[13])
iris = sns.load_dataset("iris")
#sns.jointplot("sepal_width", "petal_length", data=iris, kind="kde", space=0.2, color=tableau_20[10])
data1 = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]], size=400)
data2= np.random.multivariate_normal([0,0],[[1,-0.8],[-0.8,1]],size=200)
df1=pd.DataFrame(data1, columns=['x1','y1'])
df2 = pd.DataFrame(data2, columns=['x2','y2'])
graph=sns.jointplot(x=df1.x1, y=df1.y1, color=tableau_20[14])
graph.x=df2.x2
graph.y=df2.y2
graph.plot_joint(plt.scatter, marker='x', c=tableau_20[8], s=60)
plt.show()
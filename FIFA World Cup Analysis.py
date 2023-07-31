#!/usr/bin/env python
# coding: utf-8

# # import library

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
import warnings 
warnings.filterwarnings("ignore")


# # Read Datasset

# In[3]:


fifa22 =pd.read_csv(r'C:\Users\hp\OneDrive\Documents\Desktop\FIFA.csv')
fifa22


# In[4]:


fifa22.head()


# In[5]:


fifa22.tail()


# # View DataSet summery

# In[6]:


fifa22.info()


# In[7]:


fifa22["Body Type"].value_counts()


# # Visualize distribution of Age variable with Seaborn distplot() function

# In[8]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa22["Age"]
ax = sns.distplot(x, bins=10)
plt.show()


# # Comment
# :- It can be seen that the Age variable is slightly positively skewed.
# 
# :- We can use Pandas series object to get an informative axis label as follows-

# In[9]:


f, ax = plt.subplots(figsize=(8, 6))
x = fifa22["Age"]
x = pd.Series(x, name="Age variable")
ax = sns.distplot(x, bins=10)
plt.show()


# we can plot the distribution on the variable axis as follows:-

# In[10]:


f, ax = plt.subplots(figsize=(8, 6))
x = fifa22["Age"]
ax = sns.distplot(x, bins=10, vertical = True)
plt.show()


# # Seaborn Kernel Density Estimation (KDE) Plot

# In[11]:


f,ax = plt.subplots(figsize=(8, 6))
x = fifa22["Age"]
x = pd.Series(x, name="Age variable")
ax = sns.kdeplot(x)
plt.show()


# we can shade under the density curve and use a different color as follows:-

# In[12]:


f, ax = plt.subplots(figsize=(8, 6))
x = fifa22["Age"]
x = pd.Series(x, name="Age variable")
ax = sns.kdeplot(x, shade=True, color="r")
plt.show()


# Histogram

# In[13]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa22['Age']
ax = sns.distplot(x, kde=False, rug=True, bins=10)
plt.show()


# we can plot a kde plot alternatively as follows:-

# In[14]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa22['Age']
ax = sns.distplot(x, hist=False, rug=True, bins=10)
plt.show()


# # Explore Preferred Foot variable
# 
# # Check number of unique values in Preferred Foot variable

# In[15]:


fifa22["Preferred Foot"].nunique()


# In[16]:


fifa22["Preferred Foot"].value_counts()


# The Preferred Foot variable contains two types of values - Right and Left

# # Visualize distribution of values with Seaborn countplot() function.

# In[17]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", data=fifa22, color="c")
plt.show()


# In[18]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", hue="Real Face", data=fifa22)
plt.show()


# In[19]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y="Preferred Foot", data=fifa22, color="c")
plt.show()


# # Seaborn Catplot() function

# In[20]:


g = sns.catplot(x="Preferred Foot", kind="count", palette="ch:25", data=fifa22)


# # Explore International Reputation variable

# In[21]:


fifa22.head(1)


# In[22]:


fifa22["International Reputation"].nunique()


# In[23]:


fifa22["International Reputation"].value_counts()


# In[24]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa22)
plt.show()


# In[25]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa22, jitter=0.01)
plt.show()


# In[26]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot",data=fifa22,  palette="Set2", size=20, marker="D", edgecolor="gray", alpha=.25)
plt.show()


# In[27]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=fifa22["Potential"])
plt.show()


# In[28]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="International Reputation", y="Potential", data=fifa22)
plt.show()


# In[29]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa22, palette = "Set3")
plt.show()


# # Seaborn violinplot() function

# In[30]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x=fifa22["Potential"])
plt.show()


# In[31]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation",y="Potential", data=fifa22)
plt.show()


# In[32]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation",y="Potential",hue="Preferred Foot", data=fifa22, palette="muted")
plt.show()


# In[33]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
               data=fifa22, palette="muted", split=True)
plt.show()


# # Seaborn pointplot() function

# In[34]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", data=fifa22)
plt.show()


# In[35]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential",hue= "Preferred Foot", data=fifa22)
plt.show()


# In[36]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential",hue= "Preferred Foot", data=fifa22, dodge=True)
plt.show()


# In[37]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
              data=fifa22, markers=["o", "x"], linestyles=["-", "--"])
plt.show()


# # Seaborn barplot() function

# In[38]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa22)
plt.show()


# In[39]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential",hue= "Preferred Foot",  data=fifa22)
plt.show()


# In[40]:


from numpy import median
f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa22, estimator=median)
plt.show()


# In[41]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa22, ci=68)
plt.show()


# In[42]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential",  data=fifa22, ci="sd")
plt.show()


# In[43]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa22, capsize=0.2)
plt.show()


# # Visualizing statistical relationship with Seaborn relplot() function

# # Seaborn relplot() function
# 

# In[44]:


fifa22.head()


# In[45]:


g = sns.relplot(x="Overall", y="Potential", data=fifa22)


# # seaborn scatterplot() function

# In[46]:


f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Height", y="Weight", data=fifa22)
plt.show()


# # seaborn lineplot() function

# In[47]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.lineplot(x="Stamina", y="Strength", data=fifa22)
plt.show()


# # Seaborn regplot() function

# In[48]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa22)
plt.show()


# In[49]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa22,color= "g", marker="+")
plt.show()


# In[50]:


f, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="International Reputation", y="Potential", data=fifa22, x_jitter=.01)
plt.show()


# # seaborn lmplot() function

# In[51]:


g = sns.lmplot(x="Overall", y="Potential", data=fifa22)


# In[52]:


g = sns.lmplot(x="Overall",y="Potential", hue="Preferred Foot", data=fifa22)


# In[53]:


g = sns.lmplot(x="Overall",y="Potential",hue="Preferred Foot", data=fifa22, palette="Set1")


# In[54]:


g = sns.lmplot(x="Overall",y="Potential", col="Preferred Foot",data=fifa22)


# # multi plot grids

# # seaborn Facetgrid() function

# In[55]:


g = sns.FacetGrid(fifa22, col="Preferred Foot")


# In[56]:


g = sns.FacetGrid(fifa22, col="Preferred Foot")
g = g.map(plt.hist, "Potential")


# In[57]:


g = sns.FacetGrid(fifa22, col="Preferred Foot")
g = g.map(plt.hist, "Potential", bins=10, color="r")


# # We can plot a bivariate function on each facet as follows-

# In[58]:


g = sns.FacetGrid(fifa22,col="Preferred Foot")
g = (g.map(plt.scatter, "Height", "Weight", edgecolor="w").add_legend())
     


# In[59]:


g = sns.FacetGrid(fifa22, col="Preferred Foot", height=5, aspect=1)
g = g.map(plt.hist, "Potential")


# # Seaborn Pairgrid() function
# 

# In[61]:


fifa22_new = fifa22[["Age", "Potential", "Strength", "Stamina", "Preferred Foot"]]


# In[62]:


g = sns.PairGrid(fifa22_new)
g = g.map(plt.scatter)


# In[65]:


g = sns.PairGrid(fifa22_new)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)


# In[71]:


g = sns.PairGrid(fifa22_new, hue="Preferred Foot")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# In[72]:


g = sns.PairGrid(fifa22_new, hue="Preferred Foot")
g = g.map_diag(plt.hist, histtype="step", linewidth=3)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# In[74]:


g = sns.PairGrid(fifa22_new, vars=['Age', 'Stamina'])
g = g.map(plt.scatter)


# In[76]:


g = sns.PairGrid(fifa22_new)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, cmap="Blues_d")
g = g.map_diag(sns.kdeplot, lw=3, legend=False)


# # Seaborn `Jointgrid()` function
# 
# 
# - This function provides a grid for drawing a bivariate plot with marginal univariate plots.
# 
# - It set up the grid of subplots.

# In[79]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa22)
g = g.plot(sns.regplot, sns.distplot)


# In[80]:


import matplotlib.pyplot as plt


# In[82]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa22)
g = g.plot_joint(plt.scatter, color=".5", edgecolor="white")
g = g.plot_marginals(sns.distplot, kde=False, color=".5")


# In[84]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa22, space=0)
g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
g = g.plot_marginals(sns.kdeplot, shade=True)


# In[85]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa22, height=5, ratio=2)
g = g.plot_joint(sns.kdeplot, cmap="Reds_d")
g = g.plot_marginals(sns.kdeplot, color="r", shade=True)


# In[87]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa22);


# In[89]:


sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa22, col_wrap=2, height=5, aspect=1)


# In[90]:


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)


# In[92]:


sinplot()


# In[93]:


sns.set()
sinplot()


# In[94]:


sns.set_style("whitegrid")
sinplot()


# In[95]:


sns.set_style("dark")
sinplot()


# In[96]:


sns.set_style("white")
sinplot()


# In[97]:


sns.set_style("ticks")
sinplot()


# In[ ]:





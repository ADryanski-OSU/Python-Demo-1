# Andrew Dryanski - CWID 20247795
# MSIS 5193 - Fall 2020
# Descriptive Stats ICE
#
# Step 0: Import Packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import KFold


### Summarizing the Data (2 pts.)
#Using the data file [pollute.txt](data/pollute.txt), assess the basic descriptive information of the data. 
#Specifically, provide the mean, median, variance, standard deviation, min, max, data type, number of missing 
#    values, and the number of unique values for the following variables:
# * pollution
# * temp
# * industry
# * population
# * wind
# * rain
# * wet.days
df1 = pd.read_csv("data/pollute.txt", sep='\t', lineterminator='\n')
df1.rename(columns={
    'wet.days\r': 'wet.days'
    },
    inplace=True)
df1.head()
df1.dtypes
df1.pollution.unique()

# Stats
df1.describe(include='all')
# Print all unique values and their counts.
for i in df1:
    print(df1[i].value_counts(ascending=True))
# Print array of unique values in df1
np.unique(df1) 
np.unique(df1.temp) 




### Using Plots with Python (3 pts.)
# Using the data file [pollute.txt](data/pollute.txt), assess each variable for outliers and linearity using 
# various plots. Provide screenshots of the plots and justify in writing (within the Word file) your choice 
# of plot; do this for each plot.

# Box plots are a simple way of finding outliers.  Anything outside of the IQR can be called an outlier.
# Here, I've made a box plot of everything that shared (roughly) the same scale.
# Pollution, temperature, rain, and wet-days all out a few outliers, but nothing too wild.  Pollution is 
# obviously skewed, though.  
df1.loc[:,['pollution','temp','wind','rain','wet.days']].boxplot()

# The scatter plot also gives an immediate visual weight which can help to identify outliers and influentials.
# Here, I've put wet days ran rain together, and you can see some collinearity.
df1.plot.scatter(x='wet.days', y='rain')


# Industry and population share roughly the same scale as wll.  So I've put their box plots together.
# There are numerous outliers on Industry, and the distribution looks skewed.
df1.loc[:,['industry','population']].boxplot()

# The scatter plot of population and industry shows just how correlated the two are.  Without even
# dropping a linear model onto the plot, you can see a clear pattern that's reinforced by the extreme
# influential outliers.
df1.plot.scatter(x='industry', y='population')


# More tests would get into the variance of the error terms and such, requiring linear regression models.
# I found a tutorial to get into that and will review more in the future, but seems outisde of scope here.
# Link for my own benefit:
# https://towardsdatascience.com/verifying-the-assumptions-of-linear-regression-in-python-and-r-f4cd2907d4c0 


### Assessing Normality with Python (2 pts.)
# Using the data file [ozone.data.txt](data/ozone.data.txt), assess normality using both QQ plots and 
# Shapiro-Wilk tests for each of the following variables:
# * `rad` *solar radiation*
# * `temp` *temperature*
# * `wind` *wind speed*
# * `ozone` *ozone*
# Discuss what you see in the QQ plots and whether they agree with the results of the Shapiro-Wilk test.
df2 = pd.read_csv("data/ozone.data.txt", sep='\t', lineterminator='\n')
df2.rename(columns={
    'ozone\r': 'ozone'
    },
    inplace=True)
df2.head()
df2.dtypes
df2.describe()

# QQ plot and Shapiro-Wilk test for solar radiation
# Well defined S-curve, crossing the diagonal three times.
# Non-normal distribution, obvious outliers.  Kurtosis issues.
# Shapiro-Wilk Test for normality significant, so normality violated.
sts.probplot(df2.rad, dist="norm", plot=plt)
sts.shapiro(df2.rad)

# QQ plot and Shapiro-Wilk test for temperature
# slight s-curve.  Extreme values peeling off from the series.
# Could be a problem for normality.  May have low kurtosis.
# Shapiro-Wilk test is not significant, normality preserved.
sts.probplot(df2.temp, dist="norm", plot=plt)
sts.shapiro(df2.temp)

# QQ plot and Shapiro-Wilk test for wind 
# hugs the diagonal, little deviation, good signs for normality
# Shapiro-Wilk test is not significant, normality preserved.  
sts.probplot(df2.wind, dist="norm", plot=plt)
sts.shapiro(df2.wind)

# QQ plot and Shapiro-Wilk test for ozone
# Strong bowing, normality likely violated
# Shapiro-Wilk test is significant, so normality violated.
sts.probplot(df2.ozone, dist="norm", plot=plt)
sts.shapiro(df2.ozone)


### Skewness and Kurtosis (3 pts.)
# Using the data file [ozone.data.txt](data/ozone.data.txt), please do the following:
# * Assess the skew and kurtosis of each variable and report on what you find. 
# * Compare the values for skewness and kurtosis to histograms. 
#   * Do you find that the values correspond with what you see? 
#   * Which of the variables are left skewed? Right skewed? 
#   * How does the kurtosis look?

#print Skew and Kurtosis for each variable in the table, 
# then make one plot with all the histograms as subplots.
### It looks as though that rad and temp have some left skew, 
###  whereas wind is mostly normal and ozone is right skew.
### The kurtossis for ozone is high, and so it is heavy-tailed.
###  and on the opposite end rad is very thin-tailed and sharp.
### None of these are too extreme, however.  There's no kurtosis 4 or skew -3.
###  so apart from ozone normality may not be out of the picture.
for o in df2:
    print('Item: '+o+'; Skew: '+str(df2[o].skew())+'; Kurtosis: '+str(df2[o].kurt()) )
    print(o+' Shapiro-Wilks Test for Normality: '+str(sts.shapiro(df2[o]))+'\n')
    df2[o].plot.hist(alpha=0.5)

#Tutorial from here: https://matplotlib.org/3.1.1/gallery/statistics/hist.html
# let's print separate histograms with matplotlib ... it prints once for each 'fig'
# run from here down to -done- to get the fig to print right
fig, axs = plt.subplots(1, 4, tight_layout=True, figsize=(20,5))

# N is the count in each bin, bins is the lower-limit of the bin, play with bin size to change shape
N, bins, patches = axs[0].hist(df2.rad, bins=12)
N, bins, patches = axs[1].hist(df2.temp, bins=12)
N, bins, patches = axs[2].hist(df2.wind, bins=12)
N, bins, patches = axs[3].hist(df2.ozone, bins=12)
#Need to revisit this, figure out how to make it iterate through each subplot

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# We can also normalize our inputs by the total number of counts
axs[0].hist(df2.rad, bins=12, density=True)
axs[1].hist(df2.temp, bins=12, density=True)
axs[2].hist(df2.wind, bins=12, density=True)
axs[3].hist(df2.ozone, bins=12, density=True)

# Now we format the y-axis to display percentage
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
# -done-

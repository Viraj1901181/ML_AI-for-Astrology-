#!/usr/bin/env python
# coding: utf-8

# # TESTING THE ASTROLOGY/ZODIAC PREDICTIONS 
# 
# 
# Recently I watched a science TV show program regarding astrology. Two men were discussing the matter and one of them mentioned that it is extremely hard to validate the astrological/zodiac/horoscope predictions as they're simply too "broad". It is also known as the so-called **[Barnum Effect](https://en.wikipedia.org/wiki/Barnum_effect)**. The phenomenon occurs when individuals believe that personality descriptions apply specifically to them, while the description can actually be applied to everyone. Seems quite relevant to astrology, doesn't it?)
# 
# ![](https://inspirationfeed.com/wp-content/uploads/2020/05/Astrology-Meme-40.jpg)
# 
# So the goal of that notebook is to try testing some of the astrology claims, practice some Python data exploration tools and of course have some fun!
# 
# #### This notebook steps: 
# 
# * [Short astrology intro and my testing approach](#section-one)
# * [Dataset Review](#section-two)
# * [Divorces Data Exploration](#section-three)
# * [Zodiac Compatibity Testing](#section-four)
# * [Conclusion](#section-five)
# 
# 
# <a id="section-one"></a>
# ### Short astrology intro and my testing approach
# 
# **[Astrology](https://en.wikipedia.org/wiki/Astrology)** is a very old discipline that came to us from the 2nd millennium BC and it claims to divine information about human affairs by studying the movements and relative positions of celestial objects like planets, stars and etc.
# Despite the fact that our modern knowledge of celestial objects is quite advanced as what it was 3,000 years ago, many people still believe in it today.
# 
# Astrologists state the existence of the “zodiac signs” like Aries, Taurus, and etc. These signs can be assigned to every person based on their birth date and can predict much about the innate character of that individual.
# 
# As mentioned, the problem is that zodiac/astrology predictions of a person are hard to test. However, we aren’t giving up yet and there is at least one astrological thing that can somehow be validated and it’s called a “zodiac signs compatibility”. Essentially it means that some couples have higher (“good fit”) and lower (“bad fit”) chances for success in their marriage/love relationships depending on the compatibility of each other’s zodiac signs. Well… I wish it did work with my wife)
# 
# 
# While it is hard to test the “good fit” of a couple, the “bad fit” — is much easier. We can perfectly determine a couple’s “bad fit” and usually it is called a… divorce) So the goal is to take a divorce dataset and see what is the distribution of “bad fit” and “good fit” compatibilities.
# 
# ***My central assumption is: if astrology does work, then we should see fewer “good fit” couples and more “bad” ones.***
# 
# <a id="section-two"></a>
# ### Dataset review
# 
# I found and uploaded to Kaggle [this interesting](https://www.kaggle.com/aagghh/divorcemarriage-dataset-with-birth-dates) dataset, which is the Mexican government [official](https://datos.gob.mx/busca/dataset/registro-civil) data for the number of divorces in the city of Xalapa, Mexico and it contains approx. 4,900 divorce records. 
# 
# The beauty of this dataset is that along with the other interesting variables, it contains each partner's birth date, which is a very rare event for a public dataset. That way we can find out the partners' zodiac signs and see if they were a good or bad fit. 
# 
# Given that astrology is calling objects like planets and stars, we assume that it works irrespective of any nationalities and world places. 
# 
# So let's rock...
# 
# <a id="section-three"></a>
# ### Divorces Data exploration
# 
# Inviting the old fellas:

# In[1]:


#import modules
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

#list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The original dataset has columns in Spanish. I've created one with the English ones (divorces_2000-2015_translated.csv) and I am going to use it

# In[2]:


#to see the data items
divorce_data = pd.read_csv('C:/Users/Asus/Desktop/DS Projects/Astrology ML_AI/Divorce_marriage dataset with birth dates/divorces_2000-2015_translated.csv')

divorce_data


# We see that there are many features for both divorced partners (males and females) and that is good. While some features like divorce, marriage date, monthly income, residence, profession - might potentially be useful, but for this warm-up exercise we want to focus on the zodiac signs and for that, we would only need the birth dates which are: 
# 
# **'DOB_partner_man'**  
# **'DOB_partner_woman'**
# 
# Other features might be used in other problems/data explorations in the later notebooks. 

# In[3]:


#dropping all coloumns except partners' birth dates
divorce_data.drop(divorce_data.columns.difference(['DOB_partner_man','DOB_partner_woman']), 1, inplace=True)

divorce_data.head


# Ok, now we've got our dataset with the DOBs only that contain 4,923 divorces
# 
# But if we look into the data we see that there are some missing values. Let's run the heatmap for those values 

# In[4]:


#heatmap for missing values
sns.heatmap(divorce_data.isnull())


# Since the variable we're missing is the birth date - we cannot substitute any of these observations because we need the exact DOB for both partners in order to determine the zodiac sign
# 
# So the only way is to get rid of all missing observations

# In[5]:


#deleting all missing observations
divorce_data = divorce_data.dropna()

#cheking clean data
sns.heatmap(divorce_data.isnull())

divorce_data.head


# Cool, now we have a clean DOB coloums with 4,374 observations left, still pretty good. Let's check out their data types

# In[6]:


#cheking data types 
divorce_data.info()


# Looks like DOB is an object and given that we need to use these dates for our analysis, we need to convert them into the date format.
# 
# The problem is that the date format in the original dataset is in a DD/MM/YY, which might be quite inconvenient to work with
# 
# But since we don't really need years, as we only need day and month in order to determine the zodiac sign, we can get the required fields first and then just get rid of years

# In[7]:


#set the right format for DOB columns 
divorce_data[['DOB_partner_man', 'DOB_partner_woman']] = divorce_data[['DOB_partner_man', 'DOB_partner_woman']].apply(pd.to_datetime)

#getting the day and month separately for both partners
divorce_data['day_partner_1'], divorce_data['month_partner_1'] = divorce_data['DOB_partner_man'].apply(lambda x: x.day), divorce_data['DOB_partner_man'].apply(lambda x: x.month)
divorce_data['day_partner_2'], divorce_data['month_partner_2'] = divorce_data['DOB_partner_woman'].apply(lambda x: x.day), divorce_data['DOB_partner_woman'].apply(lambda x: x.month)

#get rid of the DOB col as we don't need years
divorce_data = divorce_data.drop(['DOB_partner_man', 'DOB_partner_woman'], axis=1)

divorce_data.head(10)


# Cool, now we only have the month and date cols that are good to go for the zodiac sign determination. 
# 
# Generally, there are 12 zodiac signs and some astrologists argue for the 13th one, but for the simplicity of this exercise, we would take the majority of astrologists' view and take the 12 signs set-up. Here they are:
# 
# 
# 
# ![](https://vietnamtimes.org.vn/stores/news_dataimages/dieulinhvnt/092020/24/16/medium/5403_Zodiacs-improved.png)
# 
# 

# Let's create a function and turn all birth dates into the actual zodiac signs

# In[8]:


# function for the zodiac sign determination

def zodiac_sign(day, month): 
    
    if month == 12: 
        return 'Sagittarius' if (day < 22) else 'Capricorn'

    elif month == 1: 
        return 'Capricorn' if (day < 20) else 'Aquarius'

    elif month == 2: 
        return 'Aquarius' if (day < 19) else 'Pisces'

    elif month == 3: 
        return 'Pisces' if (day < 21) else 'Aries'

    elif month == 4: 
        return 'Aries' if (day < 20) else 'Taurus'

    elif month == 5: 
        return 'Taurus' if (day < 21) else 'Gemini'

    elif month == 6: 
        return 'Gemini' if (day < 21) else 'Cancer'

    elif month == 7: 
        return 'Cancer' if (day < 23) else 'Leo'

    elif month == 8: 
        return 'Leo' if (day < 23) else 'Virgo'

    elif month == 9: 
        return 'Virgo' if (day < 23) else 'Libra'

    elif month == 10: 
        return 'Libra' if (day < 23) else 'Scorpio'

    elif month == 11: 
        return 'Scorpio' if (day < 22) else 'Sagittarius'


# In[9]:


#creating additional cols for zodiac signs for both partners

divorce_data['Zod_sign_partner_1'] = divorce_data.apply(lambda x: zodiac_sign(x['day_partner_1'], x['month_partner_1']), axis=1)
divorce_data['Zod_sign_partner_2'] = divorce_data.apply(lambda x: zodiac_sign(x['day_partner_2'], x['month_partner_2']), axis=1)


# In[10]:


#checking our zodiac signs do make sense
divorce_data.head(10)


# In[11]:


#once they do make sense, we can delete all other cols exept zodiac signs
divorce_data.drop(divorce_data.columns.difference(['Zod_sign_partner_1','Zod_sign_partner_2']), 1, inplace=True)

divorce_data.head(10)


# Looks cool. So now we can do some visuals on them 

# In[12]:


#plotting a chart for a number of men zodiac signs across the data
plt.figure(figsize=(12,4))
plot = sns.countplot(x="Zod_sign_partner_1", data=divorce_data, palette="PuBu").set_title('Men Zodiac Signs')
plt.xlabel("")
plt.ylabel("count of zodiacs")

#making a one for women 
plt.figure(figsize=(12,4))
plot = sns.countplot(x="Zod_sign_partner_2", data=divorce_data, palette="gist_earth").set_title('Women Zodiac Signs')
plt.xlabel("")
plt.ylabel("count of zodiacs")


# Most of the signs are quite equally distributed. However, as we're trying to test the zodiacs compatibility, we are not looking into the individual zodiac signs, but into their combinations. 
# 
# There are 12 zodiac signs for both partners and that means 144 possible combinations. So let's first look into them:

# In[13]:


#creating a matrix for the zodiac signs combinations (man+woman)
adjacency_matrix = pd.crosstab(divorce_data.Zod_sign_partner_1, divorce_data.Zod_sign_partner_2)
idx = adjacency_matrix.columns.union(adjacency_matrix.index)
adjacency_matrix = adjacency_matrix.reindex(index = idx, columns=idx, fill_value=0)
adjacency_matrix.head(12)


# In[14]:


#creating an additional col for the zodiac combinations
divorce_data['Zodiac_combinations'] = divorce_data['Zod_sign_partner_1'] + divorce_data['Zod_sign_partner_2']

#plotting a chart for combinations
plt.figure(figsize=(30,10))
plot = sns.countplot(x="Zodiac_combinations", data=divorce_data, palette="cividis_r").set_title('Zodiac Signs Combinations')
plt.xlabel("")
plt.ylabel("count of combinations")
plt.xticks(rotation=90)


# In[15]:


#see the histogram for the normal distribution
plt.figure(figsize=(12,6))
sns.distplot(divorce_data['Zodiac_combinations'].value_counts(), fit=norm);
plt.xlabel("")


# It is good that we got all of the possible 144 combinations within our dataset range and that at least tells us the size of the dataset is large enough. Only the Man+Woman (partner_1+partner_2) zodiac combinations were considered as I couldn't find any zodiac methodology in the marriage outcomes prediction that is taking an individual's sex into consideration.   
# 
# It seems like the count of different combinations ranges from 13 (least) to 42 (most) and it looks like the combinations aren't quite normally distributed with some negative skewness.
# 
# So probably there is still a chance that we get some correlations between "bad fit" combinations and divorces within this dataset. 
# 
# <a id="section-four"></a>
# ### 4. Zodiac Compatibity Testing
# 
# So how do we know a "good" or "bad" marriage/relationship fit from the astrology standpoint?
# 
# I searched that for a bit on the web and finally came up with this [website](https://numerologysign.com/astrology/zodiac/compatibility/) and this "Zodiac Compatibility Matrix":
# 
# ![](https://numerologysign.com/wp-content/uploads/2020/03/Astrological-Zodiac-Signs-Compatibility-Chart.png)
# 
# - it has a relatively high web [traffic](https://www.similarweb.com/website/numerologysign.com?utm_source=addon&utm_medium=chrome&utm_content=header&utm_campaign=cta-button&from_ext=1) with 680k+ monthly visits
# 
# - the author of that compatibility matrix is a member of the [National Council of Geocosmic Research](https://geocosmic.org/) and holds a Level II Certification from this organization. Yes, there are Councils in Astrology and they probably even get some funding...
# 
# - at the end of the day: all of these "zodiac compatibility" tables/matrix give very similar compatibilities across each other
# 
# Let's upload and plot this matrix in a .csv version:

# In[17]:


#upload the above compatibility matrix from a .csv 
comp_matrix = pd.read_csv('C:/Users/Asus/Desktop/DS Projects/Astrology ML_AI/Divorce_marriage dataset with birth dates/Comp_matrix.csv')

comp_matrix.head(10) #compatibility rate - %


# In[18]:


#making a scatter plot

plt.figure(figsize=(30,10))
x, comb = np.unique(comp_matrix['Zodiac_combination'], return_inverse=True)
plt.scatter(comb, comp_matrix['Compatibility_rate'])
plt.xticks(range(len(x)), x)
plt.xticks(rotation=90)
plt.show()


# In[19]:


#see the histogram for the normal distribution
plt.figure(figsize=(12,6))
sns.distplot(comp_matrix['Compatibility_rate'], fit=norm);
plt.xlabel("")


# In[20]:


#see the max/min, mean etc.
comp_matrix['Compatibility_rate'].describe()


# The zodiac compatibility matrix simply shows how good the zodiac signs fit each other in the range of 1 to 100% from the astrology point of view. Now we need to determine the "bad" and "good" fits, but where is the borderline for that? Given it is 1-100%, should we simply put 50%? 
# 
# From the histogram, we see that the compatibilities aren't well-distributed from 1 to 100% and they are quite positively skewed into the second 50%. Well, this is what astrology says and we have to deal with that. As seen from the describe() function - the minimum compatibility is 27% - obviously the bad fit, and the maximum compatibility of 98% is definitely a good one.  Also, the standard deviation is 0.22 and the mean is 0.67, so 0.5 (50%) isn't really a good idea here as we have many more zodiac sign combinations in the 50-100% range. 
# 
# Probably the good idea is to ask Pandas pd.qcut() function to help us. It will divide the range into equal-sized buckets based on sample quantiles (medians). In the below example I divide the range into 2 categories with one median, but this awesome function gives an option to split any dataset/range into many chunks using quantiles and multiple medians.

# In[24]:


#setting the categories
compatibility_fit_labels = ['Bad_fit', 'Good_fit']

#creating a new col for the categories
comp_matrix['Compatibility'] = pd.qcut(comp_matrix['Compatibility_rate'], q= [0, .5, 1], labels=compatibility_fit_labels)


# In[25]:


#seeing how many values in each category
comp_matrix['Compatibility'].value_counts()


# We see that the categories' aren't exactly equal, but that is fine since it's a small difference (73 vs. 71) and I think we can live with that 
# 
# So I am going to join my two data frames (comp_matrix & divorce_data) and see how the newly created compatibility column would be distributed in the actual divorce dataset as this is our end goal!

# In[26]:


#rename the col for future dataframes join
comp_matrix = comp_matrix.rename(columns={'Zodiac_combination': 'Zodiac_combinations'})

#joining the datframes
cols = ['Zodiac_combinations']
divorce_data = divorce_data.join(comp_matrix.set_index(cols), on=cols)

divorce_data


# Woohhoo! Finally we're done and we can see how the zodiac compatibilities influence the divorce...

# In[27]:


# plotting a pie chart, to see how actually zodiac compatibility is distributed across the divorce dataset in percentage

labels = ['Good_fit', 'Bad_fit']
sizes = divorce_data['Compatibility'].value_counts()
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors = colors, shadow=True, startangle=90, pctdistance=0.85)

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
fig.set_size_inches(6,6)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title('Zodiac Compatibility: Divorce Data (percentage)')
plt.show()


# In[28]:


#plotting a chart to see how zodiac compatibility is distributed across the divorce dataset in actual numbers
plt.figure(figsize=(12,4))
plot = sns.countplot(x="Compatibility", data=divorce_data, facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3)).set_title('Zodiac Compatibility: Divorce Data (actual numbers)')
plt.xlabel("")
plt.ylabel("")


# Oh well...

# In[29]:


#see the number 
divorce_data['Compatibility'].value_counts()


# Essentially, in case if astrology was right, we should have seen significantly less good fits and much more bad fits in the divorce data.
# 
# But we actually see that there is an almost perfect 50/50 distribution of "bad" and "good" fit compatibilities. It's funny that we even got a little bit more of 'good' compatibilities (50.4% vs 49.6%)
# 
# What that means is that there is an equal probability for divorced people to have a good or bad zodiac fit and it's highly unlikely the zodiac signs had any effect on the individuals' divorces. At least in that particular case with the 4374 number of divorces in Xalapa, Mexico...
# 
# Next time astrology, next time...
# 
# <a id="section-five"></a>
# ### Conclusion
# 
# That was a warm-up exercise with the aim to do some data exploration and have some fun.
# 
# **What potentially could have been wrong:**
# 
# - Find a better zodiac compatibility matrix that has a normal distribution and isn't skewed into the 50-100% range. However, this is hard as they're all very similar across various astrology resources
# 
# - We need more data. 4.3k of divorces isn't that much, but generally, it's not easy to find public marriage/divorce datasets with the individuals' DOBs
# 
# - Probably more factors should be taken into account from the astrology standpoint. Not only signs but also actual dates probably should be considered, the exact time the person was born, sex, year and etc.
# 
# - Testing and the assumptions approach is completely incorrect (feel free to comment)
# 
# Anyway hope you find the exercise entertaining and will be careful about the astrology predictions in your real-life affairs. If you like this notebook and the dataset, giving some kind of credit would be very much appreciated :) Thanks!

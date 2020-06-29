from Bio import SeqIO
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from colour import Color
from scipy.spatial import distance as dist
from sklearn import linear_model
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_ind, norm
import csv
from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes

# N=8
# E03_E03_023HH_MF20200131_hgreceovered.fsa: std = 0.001, not enough data, fits component to single peak at x=522 - CUT
# A06_A06_048HH_MF20200131_new_run3.fsa: std = 0.001, not enough data, fits component to single peak at x=773 - CUT
# B02_HGSTMF_20200114_200122_B02_2020-01-22_1.fsa: std = 13.17, not enough data, fits component to single peak at x=519 - CUT
# A03_A03_031HH_MF20200131_new.fsa: std = 123.21, low signal, poor fit - corresponds to largest standard deviation
# A04_A04_032HH_MF20200131_new_run2.fsa: std = 140.54, low signal, poor fit - corresponds to largest standard deviation
# A04_A04_032HH_MF20200131_new_run3.fsa: std = 136.61, low signal, poor fit - corresponds to largest standard deviation
# D07_HGSTMF_20200213_D07_2020-02-13_1.fsa: std = 138.24, low signal, poor fit - corresponds to largest standard deviation
# E11_HGSTMF_20200114_200122_E11_2020-01-22_1.fsa: std = 63.16, low signal, poor fit - corresponds to largest standard deviation
# F11_HGSTMF_20200114_200122_F11_2020-01-22_1.fsa: std = 51.72, low signal, poor fit - corresponds to largest standard deviation
# D11_HGSTMF_20200213_D11_2020-02-13_1.fsa: std = 65.86, low signal, poor fit - corresponds to largest standard deviation
# E01_E01_007HH_MF20200131_new.fsa: std = 84.62, low signal, poor fit - corresponds to largest standard deviation
# F11_HGSTMF_20200213_F11_2020-02-13_1.fsa: std = 56.92, low signal, poor fit - corresponds to largest standard deviation
# B07_HGSTMF_20200213_B07_2020-02-13_1.fsa: std = 48.83, low signal so broad fit - corresponds to largest standard deviation
# B11_HGSTMF_20200213_B11_2020-02-13_1.fsa: std = 43.41, low signal so broad fit - corresponds to largest standard deviation
# C03_HGSTMF_20200213_C03_2020-02-13_1.fsa: std = 49.37, low signal so broad fit - corresponds to largest standard deviation
# C07_HGSTMF_20200213_C07_2020-02-13_1.fsa: std = 47.84, low signal so broad fit - corresponds to largest standard deviation

### A07_HGSTMF_20200213_A07_2020-02-13_1.fsa: std = 50.07, low signal, poor fit
### E04_E04_024HH_MF20200131_new_run3.fsa: std = 50.61, low signal so broad fit - should be assigned low instability
### E04_E04_024HH_MF20200131_new_run4.fsa: std = 40.16, low signal so broad fit - should be assigned low instability
### A11_HGSTMF_20200213_A11_2020-02-13_1.fsa: std = 37.66, low signal so broad fit - should be assigned low instability
### C11_HGSTMF_20200213_C11_2020-02-13_1.fsa: std = 38.66, low signal so broad fit - should be assigned low instability
### F11_HGSTMF_20200213_F11_2020-02-13_1.fsa: std = 56.92, low signal so broad fit - should be assigned low instability

# the following are fine
# B05_HGSTMF_20200213_B05_2020-02-13_1.fsa: std = 38.98
# E07_HGSTMF_20200213_E07_2020-02-13_1.fsa: std = 41.52
# F07_HGSTMF_20200213_F07_2020-02-13_1.fsa: std = 44.31

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    #    setp(bp['fliers'][2], color='red')
    #    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

np.random.seed(42)

"""
#raw_data = SeqIO.read('raw data/D07_JH_190411_D07_2019-04-11_1.fsa','abi')
with open('raw data/D07_JH_190411_D07_2019-04-11_1.fsa','rb') as f:
    # 5 channels
    # our data use 2 colours
    # map marker 1000: 30 fragments in red channel of from 1000 bases, blue channel is one of interest (amplified by blue dye PCR)
    # x: fragment size
    # use red channel as reference
    for seq_record in SeqIO.parse(f,'abi'):
        d = seq_record.annotations['abif_raw']
        df = pd.DataFrame.from_dict(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        print (df.columns)
        temp = []
        treat = []
        for i in range(1,9):
            temp.append(df['DATA'+str(i)].values)
            treat.append('DATA'+str(i))
        plt.hist(temp, label=treat, bins=100, alpha=.5)
        plt.yscale('log')
        plt.legend(loc='best')
        plt.show()
"""
# distance between distributions - use basic metrics (L1 / L2 norm, mutual information)
# mixture models
# clustering - doesn't seem suitable here because not comparing between individuals, and clustering base pair expression of time seems easier to do by just comparing histograms
# anomaly detection over time t in N dimensional signal - each unit of 'Size' is a dimension - lack of sufficient repeat data to support this

data = []
hists = []
x_max = 0
names = []
"""
all_data = pd.read_csv('data.csv')
all_time = pd.read_csv('samples.csv')
for i,row in all_time.iterrows():
    time_i = row['day']
    #    if time_i==17:
    #        continue    
    if not time_i in times:
        all_files = all_time[all_time['day']==time_i]['filename'].values
        mask = [0]*len(all_data['Sample File Name'])
        for j in range(len(all_files)):
            mask += all_data['Sample File Name']==all_files[j]
        data_i = all_data[np.array(mask).astype(bool)]
        hists.append([data_i['Size'].values,data_i['Height'].values])
        times.append(time_i)
        if np.nanmax(data_i['Size'].values) > x_max:
            x_max = np.nanmax(data_i['Size'].values)
# sort data by time
hists = [row for time,row in sorted(zip(times,hists))]
times = sorted(times)
"""
#all_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/new_data/jc/data.csv')
#label_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/new_data/jc/results.csv')
#all_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/new_data/rg/data.csv')
#label_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/new_data/rg/results.csv')
all_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/paper_data/data copy.csv')
label_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/paper_data/ASOdata.csv')

label_name = 'treatment'
#label_name = 'id_mf'

treat = []
ref_ii = []
times = []
group = []
seen = []

"""
days = np.unique(label_data['day'].values)
print (days)
for i in range(len(days)):
    for j in range(2,4):
        day_i = label_data[label_data['day']==days[i]]
        day_i_0 = day_i[day_i[label_name]=='untreated']
        day_i_1 = day_i[day_i[label_name]=='treated']

        day_i_0 = day_i_0[day_i_0['biological_rep']==j]
        day_i_1 = day_i_1[day_i_1['biological_rep']==j]
        
        x_0, x_1, y_0, y_1 = [], [], [], []
        for k,row in day_i_0.iterrows():
            data_k = all_data[all_data['Sample File Name']==row['filename']]
            x_0.append(data_k['Size'].values)
            y_0.append(data_k['Height'].values)
        for k,row in day_i_1.iterrows():
            data_k = all_data[all_data['Sample File Name']==row['filename']]
            x_1.append(data_k['Size'].values)
            y_1.append(data_k['Height'].values)
        x_0 = np.array([x for y in x_0 for x in y])
        y_0 = np.array([x for y in y_0 for x in y])
        x_1 = np.array([x for y in x_1 for x in y])
        y_1 = np.array([x for y in y_1 for x in y])
        treat.append('untreated')
        hists.append([x_0,y_0])
        treat.append('treated')
        hists.append([x_1,y_1])
        times.append(days[i])
        times.append(days[i])
        ref_ii.append(np.mean(day_i_0['ii'].values))
        ref_ii.append(np.mean(day_i_1['ii'].values))
"""
for i,row in all_data.iterrows():

    #    if row['Sample File Name'][:2] == '72':
    #        continue
        
    if not row['Sample File Name'] in seen:
        seen.append(row['Sample File Name'])
        treatment = label_data[label_data['filename']==row['Sample File Name']][label_name].values
        if treatment:
            treat.append(label_data[label_data['filename']==row['Sample File Name']][label_name].values[0])
        else:
            continue
        data_i = all_data[all_data['Sample File Name']==row['Sample File Name']]
        hists.append([data_i['Size'].values,data_i['Height'].values])
        #        times.append(int(row['Sample File Name'][:2]))
        names.append(row['Sample File Name'])
        ref_ii.append(label_data[label_data['filename']==row['Sample File Name']]['ii'].values[0])
        times.append(label_data[label_data['filename']==row['Sample File Name']]['day'].values[0])
        #        times.append(0)                
        group.append(label_data[label_data['filename']==row['Sample File Name']]['biological_rep'].values[0])
        #        group.append(0)
        
        if np.nanmax(data_i['Size'].values) > x_max:
            x_max = np.nanmax(data_i['Size'].values)
            print (x_max, row['Sample File Name'])

#dic = {'saline':0, 'htt_aso':1, 'con_aso':0, 'nan':-1}
#dic = {'WT':1, 'FAN1-/-':0, '1-120':-1, '1-140':-1, '1-165':-1, '1-190':-1, '1-349':-1, '1-349+120-140del':-1, '1-590':-1, '73-349del':-1, 'D981A/R982A':-1, 'F129A':-1, 'MSH3-/-':-1, 'Q123A':-1, 'S126A':-1, 'UBZ':-1, 'Y128A':-1}
dic = {'untreated':0, 'treated':1}

treat = [dic[x] for x in treat]

print (np.array(hists).shape, np.array(ref_ii).shape, np.array(treat).shape, x_max)

###HACK
x_min = 300.
x_max = 1000.

red = Color("red")
colors = list(red.range_to(Color("blue"),len(hists)))
for i in range(len(colors)):
    colors[i] = colors[i].rgb

show_plots = 0
# FIXME: should not be zero as this could bias for skipped samples
gmm_ii = [0]*len(hists)
del_idxs = []
del_names = []
# convert to histograms
for i in range(len(hists)):

    if treat[i]==-1:
        continue
    
    temp = []
    for j in range(len(hists[i][0])):
        if not np.isnan(hists[i][0][j]):
            for k in range(int(hists[i][1][j])):
                temp.append(hists[i][0][j])
    temp = np.array(temp).flatten()
    hist, bin_edges = np.histogram(temp, bins=200, range=(0,x_max), density=True)
    # normalise histograms to area
    hists[i] = [x/np.sum(hist) for x in hist]
    #    if times[i]==2 or times[i]==3 or times[i]==17 or times[i]==39:
    if True:

        X = temp.reshape(len(temp),1)

        ### HACK
        # only fit data in this window
        X = X[X<x_max]        
        X = X[X>x_min]
        
        X = X.reshape(len(X),1)

        
        # fit models with 1-10 components
        N = np.arange(1,4)
        #        N = [8]
        models = [None for i in range(len(N))]
        for j in range(len(N)):
            models[j] = GaussianMixture(N[j]).fit(X)
        # compute the AIC and the BIC
        AIC = [m.aic(X) for m in models]
        BIC = [m.bic(X) for m in models]
        # select model with lowest BIC
        M_best = models[np.argmin(BIC)]
        
        # could instead select simplest and most likely model within tolerance
        """
        tol = 1E-1
        min_bic = np.inf
        best_idx = np.nan
        for j in range(len(BIC)):
            print (BIC[j])
            if BIC[j] < min_bic and np.abs(BIC[j]-min_bic) > tol:
                min_bic = BIC[j]
                best_idx = j
        print (np.argmin(BIC), best_idx)
        M_best = models[best_idx]
        """
        
        means = M_best.means_
        std = M_best.covariances_
        weights = M_best.weights_
        # FIXME: this won't always be true...
        # take noise component as having the largest standard deviation
        noise_idx = np.argmax(std)

        x = np.linspace(x_min, x_max, 1000)
        """
        # find main allele
        main_idx = np.nan
        peak_max = -1E9
        for j in range(len(means)):
            if means[j] > 50 and means[j] < 300:
                peak_j = np.max(norm.pdf(x, loc=means[j], scale=std[j]))
                if peak_j > peak_max:
                    peak_max = peak_j
                    main_idx = j
        if np.isnan(main_idx):
            print ('WARNING! Sample',names[i],'did not find main allele! Cutting...')
            del_idxs.append(i)
            del_names.append(names[i])
        if means[main_idx] < 100.:
            print ('WARNING! Sample',names[i],'main allele has x =',means[main_idx],'which is < 1 CAG repeat!')
        """
        # likelihood-based method to determine instability
        """
        post_prob = M_best.predict_proba(X)
        like_norm = 0
        like_inst = 0
        prob_norm = [0]*post_prob.shape[0]
        prob_inst = [0]*post_prob.shape[0]
        for j in range(post_prob.shape[1]):
            # exclude noise
            if j == noise_idx:
                continue
            elif j == main_idx:
                prob_norm += post_prob[:,j]
            else:
                prob_inst += post_prob[:,j]
        # renormalise probability
        #        for j in range(len(prob_inst)):
        #            prob_norm[j] = prob_norm[j]/(prob_norm[j]+prob_inst[j])
        #            prob_inst[j] = 1-prob_norm[j]
        prob_norm = prob_norm[prob_norm!=0]
        prob_inst = prob_inst[prob_inst!=0]
        like_norm = np.sum(np.log(prob_norm))
        like_inst = np.sum(np.log(prob_inst))
        gmm_ii[i] = like_inst / (like_inst+like_norm)
        """

        """
        std_sum = 0.
        for j in range(len(std)):
            if j != noise_idx:
                std_sum += np.sqrt(std[j][0][0])
        
        for j in range(len(means)):
            if means[j] < means[main_idx] and j != noise_idx:
                gmm_ii[i] += np.sqrt(std[j][0][0]) / std_sum
        """
        
        # find minor allele
        minor_idx = np.nan
        peak_max = -1E9
        for j in range(len(means)):
            if means[j] > x_min:
                peak_j = np.max(norm.pdf(x, loc=means[j], scale=std[j]))
                if peak_j > peak_max:
                    peak_max = peak_j
                    minor_idx = j                    
        if np.isnan(minor_idx):
            print ('WARNING! Sample',names[i],'no peak for x >',x_min,'! Setting instability to zero...')
            gmm_ii[i] = 0
            continue            
            
        if minor_idx == noise_idx:
            print ('WARNING! Sample',names[i],'did not find minor allele! Setting instability to smallest possible value...')
            gmm_ii[i] = 1E-3

            if show_plots:
                print ('Sample',names[i],'number of components',len(means))
                #            print (means[main_idx],np.sqrt(std[main_idx]))
                print (means[minor_idx],np.sqrt(std[minor_idx]))
                print (gmm_ii[i],ref_ii[i],treat[i])
                print (means)
                print (np.sqrt(std))
                fig, ax = plt.subplots()
                logprob = M_best.score_samples(x.reshape(-1, 1))
                pdf = np.exp(logprob)
                ax.plot(x, pdf, '-k')
                ax.bar(bin_edges[:-1], hist, color='grey')
                #            ax.plot([means[main_idx],means[main_idx]],[np.max(pdf),np.max(pdf)*1.5],color='green')
                ax.plot([means[noise_idx],means[noise_idx]],[0,np.max(pdf)],color='red',label='Noise')
                ax.plot([means[minor_idx],means[minor_idx]],[0,np.max(pdf)],color='green',label='Minor allele')
                ax.set_yscale('log')
                ax.set_title(names[i])
                ax.set_xlabel('x')
                ax.set_ylabel('Height')
                plt.legend()
                plt.show()
            
            continue

        # take instability index as standard deviation of minor allele
        gmm_ii[i] = np.sqrt(std[minor_idx][0][0])
        # take ii as sum of standard deviations apart from noise
        #        for j in range(len(std)):
        #            if j!=noise_idx:
        #                gmm_ii[i] += np.sqrt(std[j][0][0])
        # take instability index as sum of sum of standard deviation of minor allele and pre-major peaks
        #        gmm_ii[i] += np.sqrt(std[minor_idx][0][0]) / std_sum
        
"""

lr = linear_model.LinearRegression()
times = np.array(times).reshape(len(times),1)
lr.fit(times[1:], mean_dist[1:])
print (lr.coef_)
Y_pred = lr.predict(times[1:])
ax.plot(times[1:], Y_pred)
"""

print (len(del_idxs), del_names)

gmm_ii = np.delete(gmm_ii, del_idxs)
ref_ii = np.delete(ref_ii, del_idxs)
names = np.delete(names, del_idxs)
treat = np.delete(treat, del_idxs)
times = np.delete(times, del_idxs)
group = np.delete(group, del_idxs)

print (ttest_ind(gmm_ii[treat==0],gmm_ii[treat==1]))
print (ttest_ind(ref_ii[treat==0],ref_ii[treat==1]))

gmm_ii = np.array([(x-np.min(gmm_ii))/(np.max(gmm_ii)-np.min(gmm_ii)) for x in gmm_ii])
ref_ii = np.array([(x-np.min(ref_ii))/(np.max(ref_ii)-np.min(ref_ii)) for x in ref_ii])

with open('genimix_ii_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(gmm_ii)):
        writer.writerow([names[i],gmm_ii[i],ref_ii[i],treat[i],times[i],group[i]])

fig, ax = plt.subplots()
print (len(treat),len(gmm_ii))

A = [gmm_ii[treat==0], ref_ii[treat==0]]
B = [gmm_ii[treat==1], ref_ii[treat==1]]

# first boxplot pair
bp = boxplot(A, positions = [1, 2], widths = 0.6)
setBoxColors(bp)

# second boxplot pair
bp = boxplot(B, positions = [4, 5], widths = 0.6)
setBoxColors(bp)

# set axes limits and labels
xlim(0,7)
ax.set_xticklabels(['Untreated', 'Treated'])
#ax.set_xticklabels(['FAN1-/-', 'WT'])
ax.set_xticks([1.5, 4.5])

ax.scatter([1]*len(gmm_ii[treat==0]),gmm_ii[treat==0], facecolors='none', edgecolors='black')
ax.scatter([2]*len(ref_ii[treat==0]),ref_ii[treat==0], facecolors='none', edgecolors='black')
ax.scatter([4]*len(gmm_ii[treat==1]),gmm_ii[treat==1], facecolors='none', edgecolors='black')
ax.scatter([5]*len(ref_ii[treat==1]),ref_ii[treat==1], facecolors='none', edgecolors='black')

# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
legend((hB, hR),('GenIMix', 'ii'))
hB.set_visible(False)
hR.set_visible(False)

ax.set_ylabel('Instability index')
#ax.set_xlabel('Treatment')

#ax.set_ylim(0,1.15)

t, p = ttest_ind(gmm_ii[treat==0],gmm_ii[treat==1])
if p < 0.001:
    plt.plot([1, 1, 4, 4], [1.03, 1.07, 1.07, 1.03], lw=1.5, c='black')
    plt.text((1+4)*.5, 1.07, "***", ha='center', va='bottom', color='black')
elif p < 0.01:
    plt.plot([1, 1, 4, 4], [1.03, 1.07, 1.07, 1.03], lw=1.5, c='black')
    plt.text((1+4)*.5, 1.07, "**", ha='center', va='bottom', color='black')
elif p < 0.05:
    plt.plot([1, 1, 4, 4], [1.03, 1.07, 1.07, 1.03], lw=1.5, c='black')
    plt.text((1+4)*.5, 1.07, "*", ha='center', va='bottom', color='black')

plt.show()
"""
# max plot
#max_data = np.array(max_data)
#fig, ax = plt.subplots()
#ax.scatter(max_data[:,0], max_data[:,1])

data = np.array(data)
time = np.array(time)
df = pd.DataFrame({'time':time, 'data':data})
#df = df.sort_values(by='time')
#X = np.array([y for x in df['time'].values for y in x])
#Y = np.array([y for x in df['data'].values for y in x])
X = np.array([x for x in df['time'].values])
Y = np.array([x for x in df['data'].values])
X = X.reshape(len(X),1)
print (X,Y)

X_pred = np.unique(X)
X_pred = X_pred.reshape(len(X_pred),-1)


lr = linear_model.LinearRegression()
lr.fit(X, Y)
Y_pred = lr.predict(X_pred)
fig, ax = plt.subplots()
ax.scatter(X, [(x-110)/3 for x in Y])
ax.plot(X_pred, [(x-110)/3 for x in Y_pred])
plt.show()

#kernel = DotProduct() + WhiteKernel()
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, Y)
Y_pred, Y_sigma = gpr.predict(X_pred, return_std=True)
plt.scatter(X, Y)
plt.plot(X_pred, Y_pred)
plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
         np.concatenate([Y_pred - 1.96 * Y_sigma,
                        (Y_pred + 1.96 * Y_sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.show()
"""

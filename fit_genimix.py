from Bio import SeqIO
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from colour import Color
from scipy.spatial import distance as dist
from sklearn import linear_model
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.stats import ttest_ind, norm, ks_2samp
import csv
from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes

np.random.seed(42)

# distance between distributions - use basic metrics (L1 / L2 norm, mutual information)
# mixture models
# clustering - doesn't seem suitable here because not comparing between individuals, and clustering base pair expression of time seems easier to do by just comparing histograms
# anomaly detection over time t in N dimensional signal - each unit of 'Size' is a dimension - lack of sufficient repeat data to support this

all_data = pd.read_csv('/home/paw/Data/POND/MikeFlower/detail2.csv')

filenames = np.unique(all_data['filename'].values)

raw_data, hist_data, ref_ii, treat, time, n_bins, diff = [], [], [], [], [], [], []

for f in filenames:
    data_i = all_data[all_data['filename']==f]
    if len(np.unique(data_i['diff_n'].dropna().values)) > 1 or len(np.unique(data_i['hei.peak'].dropna().values)) > 1:
        print ('Assumption broken')
        quit()
    if not (2 in data_i['diff_n'].dropna().values or 3 in data_i['diff_n'].dropna().values or 4 in data_i['diff_n'].dropna().values):
        continue
    if data_i['hei.peak'].dropna().values < 40:
        continue
    ref_ii.append(np.unique(data_i['iictrl.h'].values[0])[0])
    treat.append(np.unique(data_i['treatment'].values[0])[0])
    time.append(np.unique(data_i['day'].values[0])[0])
    n_bins.append(len(data_i['rpt'].values[~np.isnan(data_i['rpt'].values)]))
    diff.append(np.unique(data_i['diff_n'].values[0])[0])
    temp0, temp1 = [], []
    for i in range(len(data_i['rpt'].values)):
        if not np.isnan(data_i['rpt'].values[i]):
            if data_i['h.rel'].values[i] < .2:
                continue
            temp0.append([data_i['rpt'].values[i],data_i['height'].values[i]])
            for count in range(data_i['height'].values[i]):
                temp1.append(data_i['rpt'].values[i])
    raw_data.append(np.array(temp0))
    hist_data.append(np.array(temp1))
print ('n_files',len(raw_data))

ref_ii = np.array(ref_ii)
treat = np.array(treat)
time = np.array(time)
diff = np.array(diff)

x_min, x_max = 0, 300.
gmm_ii = np.zeros(len(hist_data))
std = np.zeros(len(hist_data))
del_idxs, del_names = [], []
x = np.linspace(x_min, x_max, 1000)

for i in range(len(hist_data)):
    """
    # convert to histograms
    temp = []
    for j in range(len(hist_data[i][0])):
        if not np.isnan(hist_data[i][0][j]):
            for k in range(int(hist_data[i][1][j])):
                temp.append(hist_data[i][0][j])
    temp = np.array(temp).flatten()
    """
    ### NEW
    #    temp = hist_data[i]
    
    #    hist, bin_edges = np.histogram(temp, bins=n_bins[i], range=(0,x_max), density=True)
    # normalise histograms to area
    #    hist_data[i] = [x/np.sum(hist) for x in hist]

    raw_i = raw_data[i].reshape(len(raw_data[i]),2)
    X = hist_data[i].reshape(len(hist_data[i]),1)

    cov_type = 'full'#'spherical'
    """
    N = np.arange(1,4)
    models = [None for i in range(len(N))]
    likes = []
    for j in range(len(N)):
        #        models[j] = GaussianMixture(n_components=N[j], covariance_type=cov_type).fit(X)
        models[j] = BayesianGaussianMixture(n_components=N[j], covariance_type=cov_type).fit(X)
        likes.append(models[j].lower_bound_)    
    # compute the AIC and the BIC
    #    AIC = [m.aic(X) for m in models]
    #    BIC = [m.bic(X) for m in models]
    #    print ('BIC',BIC)
    # select model with lowest IC
    #    M_best = models[np.argmin(BIC)]
    # select model with largest likelihood
    #    M_best = models[np.argmax(likes)]
    """
    n_components = 3
    M_best = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process",
                                     weight_concentration_prior=1E2,
                                     n_components=n_components,
                                     covariance_type=cov_type,
                                     #                                     reg_covar=0,
                                     init_params='kmeans',
                                     max_iter=1000,
                                     mean_precision_prior=1E2,
                                     covariance_prior=1E0*np.eye(1),
                                     random_state=42,
    )
    M_best.fit(X)
    
    means = M_best.means_
    cov = M_best.covariances_
    weights = M_best.weights_
    print (weights)
    means_new, cov_new, weights_new = [], [], []
    for i in range(len(weights)):
        if weights[i] > 0.1:
            means_new.append(means[i])
            cov_new.append(cov[i])
            weights_new.append(weights[i])
    means = np.array(means_new)
    cov = np.array(cov_new)
    weights = np.array(weights_new)

    #    large_idx = np.argmax(means[:,0])
    #    small_idx = np.argmax(means[:,1])
    large_idx = np.argmax(means)
    small_idx = np.argmax(raw_i[:,1])

    print ('Sample',filenames[i],'number of components',len(means))

    #    print (means)
    #    print (cov)
    #    print (means[large_idx], means[small_idx])
    
    ###################################################################################################################################
    plot = 1
    if plot:
        fig, ax = plt.subplots()
        ax.set_title(treat[i])
        for j in range(len(means)):
            #            logprob = M_best.score_samples(x.reshape(-1, 1))
            #            pdf = np.exp(logprob)
            #            pdf = [x*np.max(hist[bin_edges[:-1]>x_min])/np.max(pdf) for x in pdf]
            colour = 'blue'
            linestyle = '--'
            zorder = 1

            mean = means[j]
            
            if cov_type == 'diag':
                label = '(m='+str(int(mean))+', s='+str(round(np.sqrt(cov[j][0]),1))+')'
                pdf_j = norm.pdf(x, loc=mean, scale=np.sqrt(cov[j][0]))
            elif cov_type == 'full':
                label = '(m='+str(int(mean))+', s='+str(round(np.sqrt(cov[j][0][0]),1))+')'
                pdf_j = norm.pdf(x, loc=mean, scale=np.sqrt(cov[j][0][0]))
            elif cov_type == 'spherical':
                label = '(m='+str(int(mean))+', s='+str(round(np.sqrt(cov[j]),1))+')'
                pdf_j = norm.pdf(x, loc=mean, scale=np.sqrt(cov[j]))
            elif cov_type == 'tied':
                label = '(m='+str(int(mean))+', s='+str(round(np.sqrt(cov[j][j]),1))+')'
                pdf_j = norm.pdf(x, loc=mean, scale=np.sqrt(cov[j][j]))
            pdf_j = [x*raw_i[:,1][(np.abs(raw_i[:,0] - mean)).argmin()]/np.max(pdf_j) for x in pdf_j]
            ax.plot(x, pdf_j, zorder=zorder,label=label,color=colour,linestyle=linestyle)
        ax.scatter(raw_i[:,0], raw_i[:,1])
        ax.set_xlabel('x')
        ax.set_ylabel('Height')
        plt.legend()
        
    # take instability index as standard deviation of large allele
    #        gmm_ii[i] = np.sqrt(std[large_idx][0][0])
    #        gmm_ii[i] = np.sqrt(std[large_idx][0][0]) * means[large_idx][0]
    #    gmm_ii[i] = np.sqrt(cov[large_idx][0][0]) * (means[large_idx][0]-means[small_idx][0])
    #    gmm_ii[i] = (means[large_idx][0]-means[small_idx][0]) * (means[large_idx][1]/means[small_idx][1])
    for j in range(len(means)):
        #        gmm_ii[i] += (raw_i[small_idx][0] - mean) * (raw_i[(np.abs(raw_i[:,0] - mean)).argmin()][1]/raw_i[small_idx][1])# * cov[j]
        #        gmm_ii[i] += (means[j] - raw_i[small_idx][0]) * (raw_i[(np.abs(raw_i[:,0] - means[j])).argmin()][1]/np.sum(raw_i[:][1]))
        gmm_ii[i] += (means[j] - 125.) * (raw_i[(np.abs(raw_i[:,0] - means[j])).argmin()][1]/np.sum(raw_i[:][1]))
        if j != large_idx:
            std[i] += cov[j]
#    if cov_type == 'spherical':
#        gmm_ii[i] = (raw_i[small_idx][0] - means[large_idx]) * (raw_i[(np.abs(raw_i[:,0] - means[large_idx])).argmin()][1]/raw_i[small_idx][1]) * np.sqrt(cov[large_idx])
    print (gmm_ii[i],ref_ii[i],treat[i])
    if plot:
        plt.show()

#gmm_ii = np.array([(x-np.min(gmm_ii))/(np.max(gmm_ii)-np.min(gmm_ii)) for x in gmm_ii])
#ref_ii = np.array([(x-np.min(ref_ii))/(np.max(ref_ii)-np.min(ref_ii)) for x in ref_ii])

gmm_ii_mean, gmm_ii_std, ref_ii_mean, ref_ii_std, treat_mean, time_mean = [], [], [], [], [], []
for time_i in np.unique(time):
    gmm_temp = gmm_ii[time==time_i]
    ref_temp = ref_ii[time==time_i]
    treat_temp = treat[time==time_i]
    diff_temp = diff[time==time_i]
    for diff_i in np.unique(diff_temp):
        gmm_temp_0 = gmm_temp[diff_temp==diff_i]
        ref_temp_0 = ref_temp[diff_temp==diff_i]
        treat_temp_0 = treat_temp[diff_temp==diff_i]
        for treat_i in np.unique(treat_temp_0):
            gmm_temp_1 = gmm_temp_0[treat_temp_0==treat_i]
            ref_temp_1 = ref_temp_0[treat_temp_0==treat_i]
            print (len(gmm_temp_1), len(ref_temp_1))
            gmm_ii_mean.append(np.mean(gmm_temp_1))
            gmm_ii_std.append(np.std(gmm_temp_1))
            ref_ii_mean.append(np.mean(ref_temp_1))
            ref_ii_std.append(np.std(ref_temp_1))
            treat_mean.append(treat_i)
            time_mean.append(time_i)

with open('genimix_ii_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(gmm_ii)):
        writer.writerow([filenames[i],gmm_ii[i],ref_ii[i],treat[i],time[i],std[i]])

with open('genimix_out.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(gmm_ii_mean)):
        writer.writerow([i,gmm_ii_mean[i],gmm_ii_std[i],ref_ii_mean[i],ref_ii_std[i],treat_mean[i],time_mean[i]])

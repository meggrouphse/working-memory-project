# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:14:53 2024

@author: User
"""

def statistics_accuracy(Data_path, File, Threshold):
    import pandas as pd
    import numpy as np
    import scipy as sc
    import os 
    import seaborn as sns
    import pingouin as pg
    import matplotlib
    import matplotlib.pyplot as plt
  
    
    os.chdir(Data_path)
    df  = pd.read_excel(File)
    dfc = df[df['Minimum perf'] >= Threshold]
  
    sivi = dfc['SiVi']
    sive = dfc['SiVe']
    covi = dfc['CoVi']
    cove = dfc['CoVe']
    
    hax1 = matplotlib.pyplot.hist(sivi, bins=10, alpha=0.5, range=(30,60))
    matplotlib.pyplot.title("Simple visual accuracy", {'fontsize': 20})
    plt.savefig('Simple visual accuracy.png', bbox_inches='tight')
    plt.close()
    
    hax2 = matplotlib.pyplot.hist(sive, bins=10, alpha=0.5,range=(30,60))
    matplotlib.pyplot.title("Simple verbal accuracy", {'fontsize': 20})
    plt.savefig('Simple verbal accuracy.png', bbox_inches='tight')
    plt.close()
    hax3 = matplotlib.pyplot.hist(covi, bins=10, alpha=0.5,range=(30,60))
    matplotlib.pyplot.title("Complex visual accuracy", {'fontsize': 20})
    plt.savefig('Complex visual accuracy.png', bbox_inches='tight')
    plt.close()
    hax4 = matplotlib.pyplot.hist(cove, bins=10, alpha=0.5,range=(30,60))
    matplotlib.pyplot.title("Complex verbal accuracy", {'fontsize': 20})
    plt.savefig('Complex verbal accuracy.png', bbox_inches='tight')
    plt.close()
    # MEAN AND STD
    sivi_m  = sivi.mean()
    sivi_sd = np.std(sivi)
    
    sive_m  = sive.mean()
    sive_sd = np.std(sive)
    
    covi_m  = covi.mean()
    covi_sd = np.std(covi)
    
    cove_m  = cove.mean()
    cove_sd = np.std(cove)

    #STATISTICS
    aa_sivive = pg.wilcoxon(sivi, sive, alternative='two-sided')
    ab_sicovi = pg.wilcoxon(sivi, covi, alternative='two-sided')
    ac_sicove = pg.wilcoxon(sive, cove, alternative='two-sided')
    ad_covive = pg.wilcoxon(covi, cove, alternative='two-sided')

    ##PLOT
    sivi_percent = sivi/60*100
    sive_percent = sive/60*100
    covi_percent = covi/60*100
    cove_percent = cove/60*100

    plt.figure()
    to_vis = [sivi, sive, covi, cove]
    ax = sns.boxplot(data=to_vis, x=None, y=None, hue=None, order=None, 
                    hue_order=None, orient=None, color=None, palette=None, saturation=0.75, 
                    width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5)
    ax = sns.stripplot(data = to_vis, color = 'b', size = 6)
    ax.set_xticklabels(['Simple visual','Simple verbal','Complex visual','Complex verbal'], fontsize=20)
    plt.xlabel('Experimental conditions', fontweight='bold')
    plt.ylabel('Accuracy [%]', fontweight='bold')
    ax.set_title('Accuracy across conditions')
    sns.set(font_scale=1.8) 
    
    plt.savefig('Accuracy.png', bbox_inches='tight')

    measures = {'sivi': [sivi_m, sivi_sd],
                'sive': [sive_m, sive_sd],
                'covi': [covi_m, covi_sd],
                'cove': [cove_m, cove_sd]}
    
    statistics = {'simple vi-ve': aa_sivive,
                  'visual si-co': ab_sicovi,
                  'verbal si-co': ac_sicove,
                  'complex vi-ve': ad_covive}
    
    return ax, statistics, measures


def statistics_rt(Data_path, File, Threshold):
    import pandas as pd
    import numpy as np
    import scipy as sc
    import os 
    import seaborn as sns
    import pingouin as pg
    import matplotlib
    import matplotlib.pyplot as plt
    
    os.chdir(Data_path)
    df  = pd.read_excel(File)
    dfc = df[df['Minimum perf'] >= Threshold]
  
    sivi = dfc['SiVi_RT']
    sive = dfc['SiVe_RT']
    covi = dfc['CoVi_RT']
    cove = dfc['CoVe_RT']
    
    matplotlib.pyplot.hist(sivi)
    matplotlib.pyplot.title("Simple visual RT", {'fontsize': 20})
    plt.savefig('Simple visual RT.png', bbox_inches='tight')

    matplotlib.pyplot.hist(sive)
    matplotlib.pyplot.title("Simple verbal RT", {'fontsize': 20})
    plt.savefig('Simple verbal RT.png', bbox_inches='tight')

    matplotlib.pyplot.hist(covi)
    matplotlib.pyplot.title("Complex visual RT", {'fontsize': 20})
    plt.savefig('Complex visual RT.png', bbox_inches='tight')

    matplotlib.pyplot.hist(cove)
    matplotlib.pyplot.title("Complex verbal RT", {'fontsize': 20})
    plt.savefig('Complex verbal RT.png', bbox_inches='tight')

    # MEAN AND STD
    sivi_m  = sivi.mean()
    sivi_sd = np.std(sivi)
    
    sive_m  = sive.mean()
    sive_sd = np.std(sive)
    
    covi_m  = covi.mean()
    covi_sd = np.std(covi)
    
    cove_m  = cove.mean()
    cove_sd = np.std(cove)

    #STATISTICS
    aa_sivive = pg.ttest(sivi, sive, paired=True)
    ab_sicovi = pg.ttest(sivi, covi, paired=True)
    ac_sicove = pg.ttest(sive, cove, paired=True)
    ad_covive = pg.ttest(covi, cove, paired=True)

    ##PLOT
    plt.figure()
    to_vis = [sivi, sive, covi, cove]
    ax = sns.boxplot(data=to_vis, x=None, y=None, hue=None, order=None, 
                    hue_order=None, orient=None, color=None, palette=None, saturation=0.75, 
                    width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5)
    ax = sns.stripplot(data = to_vis, color = 'b', size = 6)
    ax.set_xticklabels(['Simple visual','Simple verbal','Complex visual','Complex verbal'], fontsize=20)
    plt.xlabel('Experimental conditions', fontweight='bold')
    plt.ylabel('Response time [s]', fontweight='bold')
    ax.set_title('Response time across conditions')
    sns.set(font_scale=1.8) 
    
    plt.savefig('RT.png', bbox_inches='tight')

    measures = {'sivi': [sivi_m, sivi_sd],
                'sive': [sive_m, sive_sd],
                'covi': [covi_m, covi_sd],
                'cove': [cove_m, cove_sd]}
    
    statistics = {'simple vi-ve': aa_sivive,
                  'visual si-co': ab_sicovi,
                  'verbal si-co': ac_sicove,
                  'complex vi-ve': ad_covive}
    
    return ax, statistics, measures


def statistics_accuracy_gender(Data_path, File, Threshold, Longformatfile):
    import pandas as pd
    import numpy as np
    import scipy as sc
    import os 
    import seaborn as sns
    import pingouin as pg
    import matplotlib
    import matplotlib.pyplot as plt
    
    os.chdir(Data_path)
    df  = pd.read_excel(File)
    dfc = df[df['Minimum perf'] >= Threshold]
  
    df1 = df.loc[(df['Gender'] == 'M')]
    sivi = df1['SiVi']
    sive = df1['SiVe']
    covi = df1['CoVi']
    cove = df1['CoVe']

    # MEAN AND STD
    msivi_m  = sivi.mean()
    msivi_sd = np.std(sivi)
    
    msive_m  = sive.mean()
    msive_sd = np.std(sive)
    
    mcovi_m  = covi.mean()
    mcovi_sd = np.std(covi)
    
    mcove_m  = cove.mean()
    mcove_sd = np.std(cove)
    
    #STATISTICS
    maa_sivive = pg.wilcoxon(sivi, sive, alternative='two-sided')
    mab_sicovi = pg.wilcoxon(sivi, covi, alternative='two-sided')
    mac_sicove = pg.wilcoxon(sive, cove, alternative='two-sided')
    mad_covive = pg.wilcoxon(covi, cove, alternative='two-sided')
  
    df2 = df.loc[(df['Gender'] == 'F')]
    sivi = df2['SiVi']
    sive = df2['SiVe']
    covi = df2['CoVi']
    cove = df2['CoVe']

    # MEAN AND STD
    fsivi_m  = sivi.mean()
    fsivi_sd = np.std(sivi)
    
    fsive_m  = sive.mean()
    fsive_sd = np.std(sive)
    
    fcovi_m  = covi.mean()
    fcovi_sd = np.std(covi)
    
    fcove_m  = cove.mean()
    fcove_sd = np.std(cove)
    
    #STATISTICS
    faa_sivive = pg.wilcoxon(sivi, sive, alternative='two-sided')
    fab_sicovi = pg.wilcoxon(sivi, covi, alternative='two-sided')
    fac_sicove = pg.wilcoxon(sive, cove, alternative='two-sided')
    fad_covive = pg.wilcoxon(covi, cove, alternative='two-sided')
    
    ##PLOT
    df = pd.read_excel(Longformatfile)

    plt.figure()
    sns.boxplot(x = df['Type'], 
                y = df['Accuracy'], 
                hue = df['Gender'])

    plt.xlabel('Experimental conditions', fontweight='bold')
    plt.ylabel('Accuracy [%]', fontweight='bold')
    plt.title('Accuracy across conditions for male and female cohorts')
    sns.set(font_scale=1.8) 
    
    plt.savefig('Accuracy_gender.png', bbox_inches='tight')

    male     = {'sivi': [msivi_m, msivi_sd],
                'sive': [msive_m, msive_sd],
                'covi': [mcovi_m, mcovi_sd],
                'cove': [mcove_m, mcove_sd]}
    
    female   = {'sivi': [fsivi_m, fsivi_sd],
                'sive': [fsive_m, fsive_sd],
                'covi': [fcovi_m, fcovi_sd],
                'cove': [fcove_m, fcove_sd]}
    
    
    male_statistics   = {'simple vi-ve':  maa_sivive,
                       'visual si-co':    mab_sicovi,
                       'verbal si-co':    mac_sicove,
                       'complex vi-ve':   mad_covive}
    
    female_statistics = {'simple vi-ve':  faa_sivive,
                       'visual si-co':    fab_sicovi,
                       'verbal si-co':    fac_sicove,
                       'complex vi-ve':   fad_covive}
    
    return ax, male, female, male_statistics, female_statistics


def statistics_rt_gender(Data_path, File, Threshold, Longformatfile):
    import pandas as pd
    import numpy as np
    import scipy as sc
    import os 
    import seaborn as sns
    import pingouin as pg
    import matplotlib
    import matplotlib.pyplot as plt
    
    os.chdir(Data_path)
    df  = pd.read_excel(File)
    dfc = df[df['Minimum perf'] >= Threshold]
  
    df1 = df.loc[(df['Gender'] == 'M')]
    sivi = df1['SiVi_RT']
    sive = df1['SiVe_RT']
    covi = df1['CoVi_RT']
    cove = df1['CoVe_RT']

    # MEAN AND STD
    msivi_m  = sivi.mean()
    msivi_sd = np.std(sivi)
    
    msive_m  = sive.mean()
    msive_sd = np.std(sive)
    
    mcovi_m  = covi.mean()
    mcovi_sd = np.std(covi)
    
    mcove_m  = cove.mean()
    mcove_sd = np.std(cove)
    
    #STATISTICS
    maa_sivive = pg.ttest(sivi, sive, paired=True)
    mab_sicovi = pg.ttest(sivi, covi, paired=True)
    mac_sicove = pg.ttest(sive, cove, paired=True)
    mad_covive = pg.ttest(covi, cove, paired=True)
  
    df2 = df.loc[(df['Gender'] == 'F')]
    sivi = df2['SiVi_RT']
    sive = df2['SiVe_RT']
    covi = df2['CoVi_RT']
    cove = df2['CoVe_RT']

    # MEAN AND STD
    fsivi_m  = sivi.mean()
    fsivi_sd = np.std(sivi)
    
    fsive_m  = sive.mean()
    fsive_sd = np.std(sive)
    
    fcovi_m  = covi.mean()
    fcovi_sd = np.std(covi)
    
    fcove_m  = cove.mean()
    fcove_sd = np.std(cove)
    
    #STATISTICS
    faa_sivive = pg.ttest(sivi, sive, paired=True)
    fab_sicovi = pg.ttest(sivi, covi, paired=True)
    fac_sicove = pg.ttest(sive, cove, paired=True)
    fad_covive = pg.ttest(covi, cove, paired=True)
  
    
    ##PLOT
    df = pd.read_excel(Longformatfile)

    plt.figure()
    sns.boxplot(x = df['Type'], 
                y = df['RT'], 
                hue = df['Gender'])
 
    plt.xlabel('Experimental conditions', fontweight='bold')
    plt.ylabel('Response time [s]', fontweight='bold')
    plt.title('Response time across conditions for male and female cohorts')
    sns.set(font_scale=1.8) 

    plt.savefig('RT_gender.png', bbox_inches='tight')

    male     = {'sivi': [msivi_m, msivi_sd],
                'sive': [msive_m, msive_sd],
                'covi': [mcovi_m, mcovi_sd],
                'cove': [mcove_m, mcove_sd]}
    
    female   = {'sivi': [fsivi_m, fsivi_sd],
                'sive': [fsive_m, fsive_sd],
                'covi': [fcovi_m, fcovi_sd],
                'cove': [fcove_m, fcove_sd]}
    
    
    male_statistics   = {'simple vi-ve':  maa_sivive,
                       'visual si-co':    mab_sicovi,
                       'verbal si-co':    mac_sicove,
                       'complex vi-ve':   mad_covive}
    
    female_statistics = {'simple vi-ve':  faa_sivive,
                       'visual si-co':    fab_sicovi,
                       'verbal si-co':    fac_sicove,
                       'complex vi-ve':   fad_covive}
    
    return ax, male, female, male_statistics, female_statistics

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:13:07 2024

@author: User
"""

Data_path      = 'C:/Users/User/Desktop/Статья Рус'
File           = 'Correct_answer _ 5.0 _ 400ms_Another_sorting (based on minimum) _1104 — time.xlsx'
Longformatfile = 'sex.xlsx'
Threshold      = 43
%matplotlib qt

from Function_file import statistics_accuracy, statistics_accuracy_gender, statistics_rt, statistics_rt_gender

hax1, hax2, hax3, hax4,ax, statistics, measures = statistics_accuracy(Data_path, File, Threshold)

ax, male, female, male_statistics, female_statistics = statistics_accuracy_gender(Data_path, File, Threshold, Longformatfile)

ax, statistics, measures = statistics_rt(Data_path, File, Threshold)

ax, male, female, male_statistics, female_statistics = statistics_rt_gender(Data_path, File, Threshold, Longformatfile)

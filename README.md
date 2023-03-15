This project is done by me (Sarthak Jain) in collaboration with Professor Sandra Safo (University of Minnesota).

This is deep learning method for simultaneously integrating data from multiple sources, where these sources provide a mix of longitudinal and cross-sectional data. The proposed method has an added benefit that it provides a bootstrap mechanism for variable/feature ranking and selection. 


The original implementation of DeepIDA network can be found here: https://github.com/lasandrall/DeepIDA.git
This project is an extension of Professor Sandra's original work in DeepIDA: https://arxiv.org/abs/2111.09964
In this work, our contributions are as follows:
1) We have modified the original DeepIDA network to integrate and classify a combination of longitudinal and cross-sectional data.
2) We have proposed several feature extraction based integration-classification pipelines.
3) We have implemented and compared several methods for variable selection, preprocessing and bootstraping strategies.


In particular, we use the proposed method on the Inflammatory Bowel Disease Multiomics database (https://ibdmdb.org/) to evaluate and compare its performance to the other methods. A brief desription and the importance of our proposed method on the IBDMDB database is given as follows:
(Crohn's disease and Ulcerative Colitis are common Inflammatory Bowel Diseases (IBD). In this work, we propose a deep learning based methodology to classify multi-view    data where the different views can potentially provide a mix of longitudinal and cross-sectional information. The goal of this work is to integrate longitudinal and cross-sectional datasets to classify subjects and to determine significant variables/features. In this work, we use the metabolomics, meta-transcriptomics, host-transcriptomics and clinical data of $n=90$ subjects (obtained from the The Inflammatory Bowel Disease Multiomics Database - IBDMDB) to classify them into "disease" and "healthy" groups. The goal of this work is to integrate longitudinal and cross-sectional data to classify subjects into the two groups, and to determine molecular profiles and signatures separating the disease groups. The main contributions of this work are as follows: (i) Since the metabolomics and host-transcriptomics data is a multi-variate time-series, we use three  methods based on Euler Characteristic, Functional Principle Component Analysis and Gated Recurrent Units to condense these time-series data into one-dimensional vectors while preserving the important characteristics of the time-series; (ii) Since both the transcriptomics datasets have several thousand variables, we utilize pre-filtering methods and linear mixed models to get rid of the insignificant features; (iii) For classification, we use the DeepIDA network that uses deep neural network followed by Integrative Discriminant analysis to effectively combine the data from different sources; and compare the performance of DeepIDA with traditional classification approaches like SVM; (iv) We use bootstraping strategy along with DeepIDA to extract the top variables of each view which 
 contributed the most in the classification performance. Through this work, we identified signatures and profiles discriminating between healthy and diseased subjects, and can shed light into the etiology of IBD.)


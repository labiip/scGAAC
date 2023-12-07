# scGAAC

Our study introduces scGAAC, a clustering method for scRNA-seq data that incorporates an attention mechanism to integrate both expression and structural information. The primary contributions of the method are as follows: (i) learning cell topology through the graph attention self-encoder module, establishing a framework for extracting intrinsic biological relationships between cells without relying on statistical assumptions; (ii) efficiently combining low-dimensional expressive features learned by the self-encoder with structured representations acquired by the graph attention network, using the attention fusion module, resulting in a joint representation more conducive to clustering; (iii) optimizing learned representations and clustering objective functions in an unsupervised manner through the self-supervised module, facilitating end-to-end training of the model. Experimental results on four authentic scRNA-seq datasets demonstrate that scGAAC enhances clustering performance.

1. Data Preprocessing: Upon acquiring the scRNA-seq data, preliminary processing of the gene expression data is essential. Once the data has undergone preprocessing, it is stored in the designated data directory. To facilitate this preprocessing, utilize the preprocess.py script. Due to space constraints on GitHub, the entire dataset cannot be hosted there. Please access the detailed data file on the following website for download.

2. Generate cell graphs. We execute the calcu_graph.py file to generate the graphs required for input and store them in the specified folder.

3. Pre-training.To enhance training outcomes, we conducted pretraining, resulting in a generated pre-trained model stored in the data folder.

4. Training. Run the scGAAC.py file to train the final model.

5. Remarks: Specific data files can be obtained from the following sources:

   The scRNA-seq datasets analyzed in the current study are publicly available. Three benchmark datasets can be downloaded from the Gene Expression Omnibus (GEO) databases with accession numbers GSE45719 (Deng data), GSE65525 (Klein data), and GSE84133 (Baron-Mouse data). The Goolam data is accessible from EMBL-EBI with the accession number E-MTAB-3321.

   




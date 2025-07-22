Run "Dense_Associative_Network_Example.ipynb" and "Umap.ipynb" to reproduce the figures of the paper. The term "dense associative network" refers to the implementation of dense associative memory studied in the paper. "Umap.ipynb" is only needed for Fig. (3). Markdown cells indicate which groups of cells must be ran to reproduced each figure.

Figures were originally made using python 3.8.5 with two separate environments for "Dense_Associative_Network_Example.ipynb" and "Umap.ipynb". The environment of "Dense_Associative_Network_Example.ipynb" was set up with numpy 1.24.4, matplotlib 3.7.5, tensorflow 2.13.1 and cmasher 1.6.3. The environment of "Umap.ipynb" was set up with numpy 1.23.5, matplotlib 3.2.2, umap 0.5.7, pandas 1.5.3, datashader 0.15.2 and seaborn 0.13.2.

We set up these two different environments to avoid dependency conflicts between tensorflow 2.13.1 and umap 0.5.7. We did not try running both notebooks in the same environment.

We did not document the "get_config" and "from_config" methods of our subclasses of tensorflow classes. For more details about them, consult the tensorflow documentation (https://www.tensorflow.org/guide/keras/serialization_and_saving). The other functions and classes that are not documented are not used in the paper.

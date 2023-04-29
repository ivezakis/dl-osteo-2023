# Osteosarcoma Classification Research Code

[![DOI](https://zenodo.org/badge/614523282.svg)](https://zenodo.org/badge/latestdoi/614523282)

This repository holds the implementation code of the paper: [Deep Learning Approaches to Osteosarcoma Diagnosis and Classification: A Comparative Methodological Approach](https://doi.org/10.3390/cancers15082290).

## Requirements
This repository has been tested with:
- Python 3.10.4
- PyTorch 1.12.0
- PyTorch Lightning 1.6.3
- Torchvision 0.13.0
- Hydra 1.3.2
- Torchmetrics 0.11.3
- Numpy 1.22.3
- Pandas 1.4.2
- Pillow 9.1.0
- Scikit-Learn 1.1.0

## Citation
If you find this research useful, please cite:

```
Vezakis, I.A.; Lambrou, G.I.; Matsopoulos, G.K. Deep Learning Approaches to Osteosarcoma Diagnosis and Classification: A Comparative Methodological Approach. Cancers 2023, 15, 2290. https://doi.org/10.3390/cancers15082290 
```

```
@Article{cancers15082290,
  AUTHOR = {Vezakis, Ioannis A. and Lambrou, George I. and Matsopoulos, George K.},
  TITLE = {Deep Learning Approaches to Osteosarcoma Diagnosis and Classification: A Comparative Methodological Approach},
  JOURNAL = {Cancers},
  VOLUME = {15},
  YEAR = {2023},
  NUMBER = {8},
  ARTICLE-NUMBER = {2290},
  URL = {https://www.mdpi.com/2072-6694/15/8/2290},
  ISSN = {2072-6694},
  ABSTRACT = {Background: Osteosarcoma is the most common primary malignancy of the bone, being most prevalent in childhood and adolescence. Despite recent progress in diagnostic methods, histopathology remains the gold standard for disease staging and therapy decisions. Machine learning and deep learning methods have shown potential for evaluating and classifying histopathological cross-sections. Methods: This study used publicly available images of osteosarcoma cross-sections to analyze and compare the performance of state-of-the-art deep neural networks for histopathological evaluation of osteosarcomas. Results: The classification performance did not necessarily improve when using larger networks on our dataset. In fact, the smallest network combined with the smallest image input size achieved the best overall performance. When trained using 5-fold cross-validation, the MobileNetV2 network achieved 91% overall accuracy. Conclusions: The present study highlights the importance of careful selection of network and input image size. Our results indicate that a larger number of parameters is not always better, and the best results can be achieved on smaller and more efficient networks. The identification of an optimal network and training configuration could greatly improve the accuracy of osteosarcoma diagnoses and ultimately lead to better disease outcomes for patients.},
  DOI = {10.3390/cancers15082290}
}
```

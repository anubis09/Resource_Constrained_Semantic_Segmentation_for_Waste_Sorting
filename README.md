# Project-4B---Semantic-Segmentation-for-Waste-sorting

- [GoogleColab notebook](https://colab.research.google.com/drive/1UJa8JJCLOn21_YG3Vs__7n8fq1W_mJ5a?pli=1#scrollTo=kVvt-7FqBnoQ)
- [ArXiv publication](https://arxiv.org/abs/2310.19407)

## Abstract
This paper addresses the need for efficient waste sort-ing strategies in Materials Recovery Facilities to minimise the environmental impact of rising waste. We propose the use of resource-constrained semantic segmentation models for segmenting recyclable waste in industrial settings. Our goal is to develop models that fit within a 10MB memory constraint, suitable for edge applications with limited pro- cessing capacity. We perform the experiments on three networks: ICNet, BiSeNet (Xception39 backbone) and ENet. Given the aforementioned limitation, we implement quanti- sation and pruning techniques on the broader nets, achiev- ing predominantly positive results while marginally impacting the Mean IoU metric. Furthermore, we conduct experiments involving diverse loss functions in order to address the implicit class imbalance of the task: the outcomes indicate improvements over the commonly used Cross-entropy loss function.

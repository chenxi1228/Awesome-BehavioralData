# Awesome-BehavioralData [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository provides a curated collection of methods and papers surveyed in [***“A Survey on Behavioral Data Representation Learning”***](https://). 
All resources are systematically organized according to the same taxonomy and categorization scheme adopted in the survey.

## Tabular Data
### Papers
| Method | Paper Title | Training Paradigm | Model Architecture | Year |
|---|---|---|---|---|
|---| **Machine Learning-based Methods** |---|---|---|
| **XGBoost** | [Xgboost: A scalable tree boosting system](https://medial-earlysign.github.io/MR_Wiki/attachments/5537821/5537823.pdf) | Supervised learning | Tree | 2016 |
| **LightGBM** | [Lightgbm: A highly efficient gradient boosting decision tree](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | Supervised learning | Tree | 2017 |
| **CatBoost** | [Catboost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) | Supervised learning | Tree | 2018 |
|---|  **Deep Learning-based Methods** |---|---|---|
| **Wide&Deep** | [Wide & deep learning for recommender systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454) | Supervised learning | GLM+MLP | 2016 |
| **DeepFM** | [Deepfm: a factorization-machine based neural network for ctr prediction](https://arxiv.org/abs/1703.04247) | Supervised learning | FM+MLP | 2017 |
| **xDeepFM** | [xdeepfm: Combining explicit and implicit feature interactions for recommender systems](https://arxiv.org/pdf/1803.05170) | Supervised learning | FM+MLP+CIN | 2018 |
| **TabNN** | [Tabnn: A universal neural network solution for tabular data](https://arxiv.org/abs/2308.14129) | Supervised learning | GBDT+MLP | 2018 |
| **RLN** | [Regularization learning networks: deep learning for tabular datasets](https://proceedings.neurips.cc/paper_files/paper/2018/file/500e75a036dc2d7d2fec5da1b71d36cc-Paper.pdf) | Supervised learning | MLP | 2018 |
| **NODE** | [Neural oblivious decision ensembles for deep learning on tabular data](https://arxiv.org/abs/1909.06312) | Supervised learning | NODE | 2019 |
| **SuperTML** | [Supertml: Two-dimensional word embedding for the precognition on structured tabular data](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Sun_SuperTML_Two-Dimensional_Word_Embedding_for_the_Precognition_on_Structured_Tabular_CVPRW_2019_paper.pdf) | Supervised learning | CNN | 2019 |
| **TabNet** | [Tabnet: Attentive interpretable tabular learning](https://ojs.aaai.org/index.php/AAAI/article/download/16826/16633) | Supervised + Self-Supervised Learning | TabNet | 2019 |
| **DeepGBM** | [Deepgbm: A deep learning framework distilled by gbt for online prediction tasks](https://scholar.archive.org/work/56qy3grnbfhibnpetezm7jhzxe/access/wayback/https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf) | Supervised + Online Learning | DeepGBM | 2019 |
| **NON** | [Network on network for tabular data classification in real-world applications](https://arxiv.org/pdf/2005.10114) | Supervised learning | NON | 2020 |
| **DNF-Net** | [Dnf-net: A neural architecture for tabular data](https://arxiv.org/abs/2006.06465) | Supervised learning | DNF-Net | 2020 |
| **VIME** | [Vime: Extending the success of self-and semi-supervised learning to tabular domain](https://proceedings.neurips.cc/paper_files/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf) | Self-Supervised + Semi-Supervised Learning | VIME | 2020 |
| **TabTransformer** | [Tabransformer: Tabular data modeling using contextual embeddings](https://arxiv.org/abs/2012.06678) | Supervised + Semi-supervised Learning | Transformer | 2020 |
| **ARM-Net** | [Arm-net: Adaptive relation modeling network for structured data](https://dl.acm.org/doi/pdf/10.1145/3448016.3457321) | Supervised learning | ARM-Net | 2021 |
| **NPT** | [Self-attention between datapoints: Going beyond individual input-output pairs in deep learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/f1507aba9fc82ffa7cc7373c58f8a613-Paper.pdf) | Supervised learning | Transformer | 2021 |
| **Regularized DNNs** | [Well-tuned simple nets excel on tabular datasets](https://proceedings.neurips.cc/paper_files/paper/2021/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf) | Supervised learning | MLP | 2021 |
| **Boost-GNN** | [Boost then convolve: Gradient boosting meets graph neural networks](https://arxiv.org/abs/2101.08543) | Supervised learning | GBDT+GNN | 2021 |
| **DNN2LR** | [Dnn2lr: Interpretation-inspired feature crossing for real-world tabular data](https://arxiv.org/abs/2008.09775) | Supervised learning | DNN2LR | 2021 |
| **IGTD** | [Converting tabular data into images for deep learning with convolutional neural networks](https://www.nature.com/articles/s41598-021-90923-y.pdf) | Supervised learning | IGTD+CNN | 2021 |
| **FT-Transformer** | [Revisiting deep learning models for tabular data](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf) | Supervised learning | Transformer | 2021 |
| **SAINT** | [Saint: Improved neural networks for tabular data via row attention and contrastive pre-training](https://arxiv.org/abs/2106.01342) | Supervised + Self-Supervised Learning | SAINT | 2021 |
| **SCARF** | [Scarf: Self-supervised contrastive learning using random feature corruption](https://arxiv.org/abs/2106.15147) | Self-Supervised Learning | MLP | 2021 |
| **GANDALF** | [Gandalf: gated adaptive network for deep automated learning of features](https://arxiv.org/abs/2207.08548) | Supervised learning | GFLU | 2022 |
| **TabDDPM** | [Tabddpm: Modelling tabular data with diffusion models](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf) | Unsupervised Learning | Diffusion Model | 2022 |
| **PTab** | [Ptab: Using the pre-trained language model for modeling tabular data](https://arxiv.org/abs/2209.08060) | Supervised + Self-Supervised Learning | BERT | 2022 |
| **TabCBM** | [TabCBM: Concept-based Interpretable Neural Networks for Tabular Data](https://openreview.net/pdf?id=TIsrnWpjQ0) | Supervised Learning | MLP | 2024 |
| **Trompt** | [Prompt: Towards a better deep neural network for tabular data](https://arxiv.org/abs/2305.18446) | Supervised + Prompt learning | MLP | 2023 |
| **HYTREL** | [Hytrel: Hypergraph-enhanced tabular data representation learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/66178beae8f12fcd48699de95acc1152-Paper-Conference.pdf) | Supervised + Self-Supervised Learning | HYTREL | 2023 |
| **ReConTab** | [Recontab: Regularized contrastive representation learning for tabular data](https://arxiv.org/abs/2310.18541) | Self-Supervised | Transformer | 2023 |
| **XTab** | [Xtab: Cross-table pretraining for tabular transformers](https://arxiv.org/abs/2305.06090) | Supervised + Self-Supervised Learning | Transformer | 2023 |
| **IATTN** | [Interpretable Additive Tabular Transformer Networks](https://openreview.net/pdf?id=TdJ7lpzAkD) | Supervised Learning | Transformer | 2024 |
| **MambaTab** | [Mambatab: A plug-and-play model for learning tabular data](https://pmc.ncbi.nlm.nih.gov/articles/PMC11755428/) | Supervised learning | Mamba | 2024 |
| **GCondNet** | [GCondNet: A Novel Method for Improving Neural Networks on Small High-Dimensional Tabular Data](https://openreview.net/pdf?id=y0b0H1ndGQ) | Supervised learning | MLP | 2024 |
| **BiSHop** | [Bishop: Bi-directional cellular learning for tabular data with generalized sparse modern hopfield model](https://arxiv.org/abs/2404.03830) | Supervised learning | BiSHop | 2024 |
| **LF-transformer** | [Lf-transformer: Latent factorizer transformer for tabular learning](https://ieeexplore.ieee.org/iel7/6287639/6514899/10401112.pdf) | Supervised learning | Transformer | 2024 |
| **TabR** | [Tabr: Tabular deep learning meets nearest neighbors](https://arxiv.org/pdf/2307.14338) | Supervised learning | TabR | 2024 |
| **TP-BERTa** | [Making pre-trained language models great on tabular prediction](https://arxiv.org/abs/2403.01841) | Supervised + Self-Supervised Learning | BERT | 2024 |
| **CARTE** | [Carte: pretraining and transfer for tabular learning](https://arxiv.org/abs/2402.16785) | Supervised + Self-Supervised Learning | Transformer | 2024 |
| **SwitchTab** | [Switchtab: Switched autoencoders are effective tabular learners](https://ojs.aaai.org/index.php/AAAI/article/download/29523/30869) | Self-Supervised Learning | Transformer | 2024 |
| **DP-2Stage** | [DP-2Stage: Adapting Language Models as Differentially Private Tabular Data Generators](https://openreview.net/pdf?id=6nBIweDYzZ) | Supervised Learning | Transformer | 2025 |
| **TARTE** | [Table Foundation Models: on knowledge pre-training for tabular learning](https://openreview.net/pdf?id=QV4P8Csw17) | Supervised + Self-Supervised Learning | Transformer | 2025 |
| **TDColER** | [On Learning Representations for Tabular Data Distillation](https://openreview.net/pdf?id=GXlsrvOGIK) | Supervised + Unsupervised Learning | Distillation | 2025 |
|---| **LLM-driven Methods** |---|---|---|
| **TAPAS** | [Tapas: Weakly supervised table parsing via pre-training](https://arxiv.org/abs/2004.02349) | Self-Supervised Learning | Bert | 2020 |
| **TAPEX** | [Tapex: Table pre-training via learning a neural sql executor](https://arxiv.org/abs/2107.07653) | Supervised + Self-Supervised Learning | Transformer | 2022 |
| **TabLLM** | [Tabllm: Few-shot classification of tabular data with large language models](https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf) | Supervised + Self-Supervised Learning | Transformer | 2022 |
| **cTBLS** | [ctbls: Augmenting large language models with conversational tables](https://arxiv.org/abs/2303.12024) | Supervised learning | Transformer | 2023 |
| **TST-LLM** | [LLM-Guided Self-Supervised Tabular Learning With Task-Specific Pre-text Tasks](https://openreview.net/pdf?id=jXcx2oAIbw) | Self-Supervised learning | Transformer | 2025 |
| **Tabby** | [Tabby: A Language Model Architecture for Tabular and Structured Data Synthesis](https://openreview.net/pdf?id=b9FPVnb0Bn) | Supervised learning | Transformer | 2026 |

### Benchmarks

| Benchmark | Paper | Repository | Year |
|---|---|---|:---:|
| OpenGL-CC18 | [arXiv](<https://arxiv.org/abs/1708.03731>) | [GitHub](<https://www.openml.org/search?type=benchmark&sort=tasks%20INCLUDED&study_type=task&id=99>) | 2017 |
| WellTunedSimpleNets | [NeurIPS](<https://proceedings.neurips.cc/paper_files/paper/2021/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf>) | [GitHub](<https://github.com/machinelearningnuremberg/WellTunedSimpleNets>) | 2021 |
| TabularBench | [NeurIPS](<https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf>) | [GitHub](<https://github.com/LeoGrin/tabular-benchmark>) | 2022 |
| TabZilla | [NeurIPS](<https://proceedings.neurips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf>) | [GitHub](<https://github.com/naszilla/tabzilla>) | 2023 |
| OpenTabs | [arXiv](<https://arxiv.org/pdf/2307.04308>) | [GitHub](<https://github.com/Chao-Ye/CM2>) | 2024 |
| TALENT | [arXiv](<https://arxiv.org/abs/2407.00956>) | [GitHub](<https://github.com/LAMDA-Tabular/TALENT>) | 2024 |
| TDBench | [TMLR](<https://openreview.net/pdf?id=GXlsrvOGIK>) | [GitHub](<http://github.com/inwonakng/tdbench>) | 2025 |


### Dataset Resources

[TabularBench-GitHub](https://github.com/LeoGrin/tabular-benchmark)

[TabularBench-Huggingface](https://huggingface.co/datasets/inria-soda/tabular-benchmark)


## Event Sequence

### Papers
| Method | Paper Title | Training Paradigm | Model Architecture | Year |
|---|---|---|---|---|
|---| **Symbolic Modeling Methods** |---|---|---|
| **FPMC** | [Factorizing personalized markov chains for next-basket recommendation](http://www.ambuehler.ethz.ch/CDstore/www2010/www/p811.pdf) | Self-supervised learning | Markov Chain | 2010 |
| **PSPM** | [Effective next-items recommendation via personalized sequential pattern mining](https://www.researchgate.net/profile/Xiaoli-Li-20/publication/237053344_Effective_Next-Items_Recommendation_via_Personalized_Sequential_Pattern_Mining/links/00b4952b7da2d4d0e7000000/Effective-Next-Items-Recommendation-via-Personalized-Sequential-Pattern-Mining.pdf) | Self-supervised learning | Sequential Pattern Mining | 2012 |
| **PRME** | [Personalized ranking metric embedding for next new poi recommendation](https://www.academia.edu/download/86224049/293.pdf) | Self-supervised learning | Markov Chain | 2015 |
|---|**Deep Learning-based Methods** |---|---|---|
| **GRU4Rec** | [Session-based recommendations with recurrent neural networks](https://arxiv.org/abs/1511.06939) | Supervised learning | GRU | 2015 |
| **RMTPP** | [Recurrent marked temporal point processes: Embedding event history to vector](https://dunan.github.io/pdf/DuDaiTriUpa2016.pdf) | Self-supervised learning | RNN | 2016 |
| **NHP** | [The neural hawkes process: A neurally selfmodulating multivariate point process](https://proceedings.neurips.cc/paper_files/paper/2017/file/6463c88460bd63bbe256e495c63aa40b-Paper.pdf) | Self-supervised learning | LSTM | 2017 |
| **Event2Vec** | [Event2vec: Learning representations of events on temporal sequences](https://www.researchgate.net/profile/Shenda-Hong/publication/318857021_Event2vec_Learning_Representations_of_Events_on_Temporal_Sequences/links/5de91e13299bf10bc34357a3/Event2vec-Learning-Representations-of-Events-on-Temporal-Sequences.pdf) | Supervised learning | Transformer | 2017 |
| **IRGAN** | [Irgan: A minimax game for unifying generative and discriminative information retrieval models](https://arxiv.org/pdf/1705.10513) | Semi-supervised learning | GAN | 2017 |
| **HRNN** | [Personalizing session-based recommendations with hierarchical recurrent neural networks](https://arxiv.org/pdf/1706.04148) | Supervised learning | RNN | 2017 |
| **Caser** | [Personalized top-n sequential recommendation via convolutional sequence embedding](https://arxiv.org/pdf/1809.07426) | Supervised learning | CNN | 2018 |
| **AttRec** | [Next item recommendation with self-attention](https://arxiv.org/abs/1808.06414) | Supervised learning | Transformer | 2018 |
| **MANN** | [Sequential recommendation with user memory networks](https://dl.acm.org/doi/pdf/10.1145/3159652.3159668) | Supervised learning | Memory Network | 2018 |
| **RecGAN** | [Recgan: recurrent generative adversarial networks for recommendation systems](https://homangab.github.io/papers/recgan.pdf) | Semi-supervised learning | GAN+RNN | 2018 |
| **BERT4Rec** | [Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer](https://arxiv.org/pdf/1904.06690) | Self-supervised learning | BERT | 2019 |
| **DTCDR** | [Dtcdr: A framework for dual-target cross-domain recommendation](https://www.researchgate.net/profile/Feng-Zhu-59/publication/337018321_DTCDR_A_Framework_for_Dual-Target_Cross-Domain_Recommendation/links/5e60a6c3a6fdccbeba1c9d86/DTCDR-A-Framework-for-Dual-Target-Cross-Domain-Recommendation.pdf) | Supervised learning | MLP | 2019 |
| **FDSA** | [Feature-level deeper self-attention network for sequential recommendation](https://www.ijcai.org/proceedings/2019/0600.pdf) | Supervised learning | Transformer | 2019 |
| **NextItNet** | [A simple convolutional generative network for next item recommendation](https://arxiv.org/pdf/1808.05163) | Supervised learning | CNN | 2019 |
| **SAHP** | [Self-attentive hawkes process](http://proceedings.mlr.press/v119/zhang20q/zhang20q.pdf) | Self-supervised learning | Transformer | 2020 |
| **THP** | [Transformer hawkes process](http://proceedings.mlr.press/v119/zuo20a/zuo20a.pdf) | Self-supervised learning | Transformer | 2020 |
| **BEHRT** | [Behrt: transformer for electronic health records](https://www.nature.com/articles/s41598-020-62922-y.pdf) | Self-supervised learning | BERT | 2020 |
| **TiSASRec** | [Time interval aware self-attention for sequential recommendation](https://dl.acm.org/doi/pdf/10.1145/3336191.3371786) | Supervised learning | Transformer | 2020 |
| **RAPT** | [Rapt: Pre-training of time-aware transformer for learning robust healthcare representation](https://drive.google.com/file/d/1YtjW7wmKoZ0nXApDZGTXK-1Otey_bhlC/view) | Self-supervised learning | Transformer | 2021 |
| **CoSeRec** | [Contrastive self-supervised sequential recommendation with robust augmentation](https://arxiv.org/abs/2108.06479) | Self-supervised learning | GAN+CL | 2021 |
| **ASReP** | [Augmenting sequential recommendation with pseudo-prior items via reversely pre-training transformer](https://dl.acm.org/doi/pdf/10.1145/3404835.3463036) | Supervised learning | Transformer | 2021 |
| **UniSRec** | [Towards universal sequence representation learning for recommender systems](https://arxiv.org/pdf/2206.05941) | Self-supervised learning | BERT+Transformer | 2022 |
| **RecGURU** | [Recguru: Adversarial learning of generalized user representations for cross-domain recommendation](https://arxiv.org/pdf/2111.10093) | Self-supervised learning | Transformer | 2022 |
| **promptTPP** | [Prompt-augmented temporal point process for streaming event sequence](https://proceedings.neurips.cc/paper_files/paper/2023/file/3c129892b4f9c8326aba665425a470c5-Paper-Conference.pdf) | Continual learning | Transformer+ Prompts | 2023 |
| **Meta TPP** | [Meta temporal point processes](https://arxiv.org/abs/2301.12023) | Meta learning | Transformer | 2023 |
| **BERT4ETH** | [Bert4eth: A pretrained transformer for ethereum fraud detection](https://dl.acm.org/doi/pdf/10.1145/3543507.3583345) | Self-supervised learning | BERT | 2023 |
| **PrimeNet** | [Primenet: Pre-training for irregular multivariate time series](https://ojs.aaai.org/index.php/AAAI/article/view/25876/25648) | Self-supervised learning | Transformer | 2023 |
| **ECGAN-Rec** | [Enhancing sequential recommendation with contrastive generative adversarial network](https://www.sciencedirect.com/science/article/pii/S0306457323000687) | Semi-supervised learning | GAN | 2023 |
| **Player2Vec** | [player2vec: A language modeling approach to understand player behavior in games](https://arxiv.org/abs/2404.04234) | Self-supervised learning | BERT | 2024 |
| **Residual TPP** | [Residual TPP: A unified lightweight approach for event stream data analysis](https://openreview.net/pdf?id=AUkBFMtyUs) | Self-supervised learning | Hawkes + Neural TPP | 2025 |
| **IOCLRc** | [Intent oriented contrastive learning for sequential recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33390/35545) | Self-supervised learning | Transformer+CL | 2025 |
| **HORAE** | [Horae: Temporal multi-interest pre-training for sequential recommendation](https://dl.acm.org/doi/abs/10.1145/3727645) | Self-supervised learning | Transformer | 2025 |

### Benchmarks
| Benchmark | Paper | Repository | Year |
|---|---|---|:---:|
| GRU | [arXiv](<https://arxiv.org/abs/1511.06939>) | [GitHub](<https://github.com/clientGe/Sequential_Recommendation_Tensorflow>) | 2015 |
| BERT | [ACL](<https://aclanthology.org/N19-1423.pdf>) | [GitHub](<https://github.com/google-research/bert>) | 2017 |
| CPC | [arXiv](<https://arxiv.org/abs/1807.03748>) | [GitHub](<https://github.com/davidtellz/contrastive-predictive-coding>) | 2018 |
| Transformer | [NeurIPS](<https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>) | [GitHub](<https://github.com/jadore801120/attention-is-all-you-need-pytorch>) | 2017 |
| PrimeNet | [AAAI](<https://ojs.aaai.org/index.php/AAAI/article/view/25876/25648>) | [GitHub](<https://github.com/ranakroychowdhury/PrimeNet>) | 2023 |
| RMTPP | [KDD](<https://dunan.github.io/pdf/DuDaiTriUpa2016.pdf>) | [GitHub](<https://github.com/ivan-chai/hotpp-benchmark>) | 2016 |


### Dataset Resources

[Recommender Systems and Personalization Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)

[MovieLens](https://grouplens.org/datasets/movielens/)

[UserBehavior](https://tianchi.aliyun.com/dataset/649)

## Dynamic Graph

### Papers
#### Discrete-Time Dynamic Graph (DTDG)
| Method | Paper Title | Structural Encoding | Temporal Encoding | Year |
|---|---|---|---|---|
|---| **Static Embedding + Temporal Alignment Methods**  |---|---|---|
| **Chakrabarti et al.** | [Evolutionary clustering](http://faculty.mccombs.utexas.edu/deepayan.chakrabarti/mywww/papers/kdd06-evolutionary.pdf) | Matrix factorization | Smoothness regularization or alignment | 2006 |
| **Chi et al.** | [On evolutionary spectral clustering](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/09tkdd_evolutionary-1.pdf) | Matrix factorization | Smoothness regularization or alignment | 2009 |
| **Kim & Han** | [A particle-and-density based evolutionary clustering method for dynamic networks](http://www.vldb.org/pvldb/vol2/vldb09-404.pdf) | Matrix factorization | Smoothness regularization or alignment | 2009 |
| **Gupta et al.** | [Evolutionary clustering and analysis of bibliographic networks](https://charuaggarwal.net/asonam-cluster.pdf) | Matrix factorization | Smoothness regularization or alignment | 2011 |
| **Yao et al.** | [Link prediction based on common-neighbors for dynamic social network](https://www.sciencedirect.com/science/article/pii/S1877050916301259) | Matrix factorization | Smoothness regularization or alignment | 2016|
| **Zhou et al.** | [Dynamic network embedding by modeling triadic closure process](https://ojs.aaai.org/index.php/AAAI/article/view/11257) | Matrix factorization | Smoothness regularization or alignment | 2018 |
| **Hisano** | [Semi-supervised graph embedding approach to dynamic link prediction](https://arxiv.org/pdf/1610.04351) | Matrix factorization | Time window aggregation | 2018 |
| **Sharan & Neville** | [Temporal-relational classifiers for prediction in evolving domains](https://www.umangsh.com/static/mysite/docs/icdm08_slides.3d3d47831d62.pdf) | Matrix factorization | Time-weighted adjacency matrices | 2008 |
| **Ibrahim et al.** | [Link prediction in dynamic social networks by integrating different types of information](https://www.researchgate.net/profile/Nahla-Ibrahim-3/publication/275219479_Link_prediction_in_dynamic_social_networks_by_integrating_different_types_of_information/links/63189822873eca0c006c48f2/Link-prediction-in-dynamic-social-networks-by-integrating-different-types-of-information.pdf) | Matrix factorization | Exponential decay | 2015 |
| **Ahmed et al.** | [Sampling-based algorithm for link prediction in temporal networks](https://www.sciencedirect.com/science/article/pii/S0020025516308507) | Low-rank adjacency | Temporal sampling strategies | 2016 |
| **Singer et al.** | [Node embedding over temporal graphs](https://arxiv.org/abs/1903.08889) | Random walk | Init. from previous step + fine-tuning | 2019 |
| **DynGEM** | [Dyngem: Deep embedding method for dynamic graphs](https://arxiv.org/abs/1805.11273) | Deep autoencoder | Regularization across snapshots | 2018 |
| **DynamicTriad** | [Dynamic network embedding by modeling triadic closure process](https://ojs.aaai.org/index.php/AAAI/article/view/11257/11116) | Triadic closure | Temporal smoothness | 2018 |
|---| **GNN+RNN-based Methods** |---|---|---|
| **GCRN** | [Structured sequence modeling with graph convolutional recurrent networks](https://arxiv.org/pdf/1612.07659) | GCN | LSTM | 2018 |
| **Narayan & Roe** | [Learning graph dynamics using deep neural networks](https://www.sciencedirect.com/science/article/pii/S2405896318300788) | GraphSAGE | LSTM | 2018 |
| **TGCN** | [T-gcn: A temporal graph convolutional network for traffic prediction](https://arxiv.org/pdf/1811.05320) | GCN | GRU | 2019 |
| **TNA** | [Temporal neighbourhood aggregation: Predicting future links in temporal graphs via recurrent variational graph convolutions](https://arxiv.org/pdf/1908.08402) | GCN | GRU | 2019 |
| **VGRNN** | [Variational graph recurrent neural networks](https://proceedings.neurips.cc/paper/2019/file/a6b8deb7798e7532ade2a8934477d3ce-Paper.pdf) | VGAE | LSTM | 2019 |
| **LRGCN** | [Predicting path failure in time-evolving graphs](https://arxiv.org/pdf/1905.03994) | R-GCN | LSTM | 2019 |
| **E-LSTM-D** | [E-lstm-d: A deep learning framework for dynamic network link prediction](https://arxiv.org/pdf/1902.08329) | Autoencoder | LSTM | 2019 |
| **EvolveGCN** | [Evolvegn: Evolving graph convolutional networks for dynamic graphs](https://aaai.org/ojs/index.php/AAAI/article/view/5984/5840) | GCN | GRU | 2020 |
| **dygraph2vec** | [dyngraph2vec: Capturing network dynamics using dynamic graph representation learning](https://arxiv.org/pdf/1809.02657) | graph2vec | LSTM/GRU | 2020 |
| **TeMP** | [Temp: Temporal message passing for temporal knowledge graph completion](https://arxiv.org/pdf/2010.03526) | GCN | GRU or Attention | 2020 |
| **WD-GCN/CD-GCN** | [Dynamic graph convolutional networks](https://arxiv.org/pdf/1704.06199) | GCN | Modified LSTM | 2020 |
| **HDGNN** | [A heterogeneous dynamical graph neural networks approach to quantify scientific impact](https://arxiv.org/abs/2003.12042) | Heterogeneous random walk | Bi-RNN | 2020 |
| **HTGN** | [Discretetime temporal network embedding via implicit hierarchical learning in hyperbolic space](https://arxiv.org/pdf/2107.03767) | Hyperbolic attention-based GCN | Hyperbolic GRU | 2021 |
| **GC-LSTM** | [Gc-lstm: Graph convolution embedded LSTM for dynamic network link prediction](https://arxiv.org/pdf/1812.04206) | GCN | LSTM | 2022 |
| **ROLAND** | [Roland: graph learning framework for dynamic graphs](https://arxiv.org/pdf/2208.07239) | GCN | Adaptive RNN | 2022 |
| **RPC** | [Learn from relational correlations and periodic events for temporal knowledge graph reasoning](https://scholar.archive.org/work/d3dqwybhbjf4flpwu76yje3cq4/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3539618.3591711) | GNN | GRU | 2023 |
| **SEIGN** | [Seign: A simple and efficient graph neural network for large dynamic graphs](https://leichuan.github.io/files/icde23-slides.pdf) | GCN-like message passing | GRU parameter adjustments | 2023 |
| **RETIA** | [Retia: relation-entity twin-interact aggregation for temporal knowledge graph extrapolation](https://opus.lib.uts.edu.au/bitstream/10453/166395/3/RETIA%20relation-entity%20twin-interact%20aggregation%20for%20temporal%20knowledge%20graph%20extrapolation.pdf) | GCN | GRU + LSTM | 2023 |
| **MegaCRN** | [Spatio-temporal meta-graph learning for traffic forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/25976/25748) | Meta-graph learner | Custom GRU | 2023 |
| **DEFT** | [Learnable spectral wavelets on dynamic graphs to capture global interactions](https://ojs.aaai.org/index.php/AAAI/article/view/25831/25603) | GNN | RNN-based parameter evolution +Wavelet | 2023 |
| **STGNPP** | [Spatio-temporal graph neural point process for traffic congestion event prediction](https://ojs.aaai.org/index.php/AAAI/article/download/26669/26441) | GCN +Transformer | Continuous GRU | 2023 |
| **WinGNN** | [Wingnn: dynamic graph neural networks with random gradient aggregation window](https://dl.acm.org/doi/pdf/10.1145/3580305.3599551) | GNN | Sliding window | 2023 |
| **SpikeNet** | [Scaling up dynamic graph representation learning via spiking neural networks](https://ojs.aaai.org/index.php/AAAI/article/view/26034/25806) | GNN | SSN | 2023 |
| **TTGCN** | [K-truss based temporal graph convolutional network for dynamic graphs](https://proceedings.mlr.press/v222/li24d/li24d.pdf) | Truss-based GCN | GRU | 2024 |
|---| **Attention-based Methods** |---|---|---|
| **DySAT** | [Dysat: Deep neural representation learning on dynamic graphs via self-attention networks](https://aravindsankar28.github.io/files/DySAT-WSDM2020.pdf) | Graph attention | Graph attention | 2020 |
| **TEDIC** | [Tedic: Neural modeling of behavioral patterns in dynamic social interaction networks](https://par.nsf.gov/servlets/purl/10300282) | Graph diffusion | Temporal Convolutional Network | 2021 |
| **DyHATR** | [Modeling dynamic heterogeneous network for link prediction using hierarchical attention with temporal mn](https://arxiv.org/pdf/2004.01024) | Hierarchical attention | Temporal attentive RNN | 2021 |
| **DREAM** | [Dream: Adaptive reinforcement learning based on attention mechanism for temporal knowledge graph reasoning](https://arxiv.org/pdf/2304.03984) | Attention | Attention + Reinforcement learning | 2023 |
| **STGNP** | [Graph neural processes for spatio-temporal extrapolation](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) | Dilated Causal Convolution | Cross-set Graph Convolution | 2023 |
| **DTFormer** | [Dtformer: A transformer-based method for discrete-time dynamic graph representation learning](https://arxiv.org/pdf/2407.18523) | Transformer | Transformer | 2024 |


#### Continuous-Time Dynamic Graph (CTDG)
| Method | Paper Title | Structure Encoding | Temporal Encoding | Memory-based | Year |
|---|---|---|---|---|---|
|---| **RNN-based Methods** |---|---|---|
| **DeepCoevolve** | [Deep coevolutionary network: Embedding user and item features for recommendation](https://arxiv.org/abs/1609.03675) | Implicit (via sequential interactions) | RNN | No | 2018 |
| **JODIE** | [Predicting dynamic embedding trajectory in temporal interaction networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC6752886/pdf/nihms-1047384.pdf) | Implicit (via sequential interactions) | RNN + Projection | No | 2020 |
| **Know-Evolve** | [Know-evolve: Deep temporal reasoning for dynamic knowledge graphs](http://proceedings.mlr.press/v70/trivedi17a/trivedi17a.pdf) | Implicit (via sequential interactions) | RNN | No | 2017 |
| **RE-Net** | [Recurrent event network: Autoregressive structure inferenceover temporal knowledge graphs](https://aclanthology.org/2020.emnlp-main.541.pdf) | GCN | RNN | No | 2020 |
| **HierTCN** | [Hierarchical temporal convolutional networks for dynamic recommender systems](https://arxiv.org/pdf/1904.04381) | Implicit (via sequential interactions) | GRU + TCN | No | 2019 |
| **DynGESN** | [Dynamic graph echo state networks](https://arxiv.org/abs/2110.08565) | Implicit (via sequential interactions) | Echo State Network | No | 2021 |
| **DyGNN** | [Streaming graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3397271.3401092) | GNN | LSTM | No | 2020 |
| **AER-AD** | [Anonymous edge representation for inductive anomaly detection in dynamic bipartite graph](https://www.vldb.org/pvldb/vol16/p1154-fang.pdf) | Local anonymous subgraph | GRU | No | 2023 |
| **RTRGN** | [Recurrent temporal revision graph networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/dafd116ac8c735f149558b79fd48e090-Paper-Conference.pdf) | Implicit (via sequential interactions) | RNN | No | 2023 |
| **TGN** | [Temporal graph networks for deep learning on dynamic graphs](https://arxiv.org/abs/2006.10637) | Implicit (via message passing) | RNN + Memory | Yes | 2020 |
| **NAT** | [Neighborhood-aware scalable temporal network representation learning](https://proceedings.mlr.press/v198/luo22a/luo22a.pdf) | Implicit (via sequential interactions) | RNN | Yes | 2022 |
| **GDCF** | [Generic and dynamic graph representation learning for crowd flow modeling](https://ojs.aaai.org/index.php/AAAI/article/download/25548/25320) | Spatiotemporal GNN | RNN + Memory | Yes | 2023 |
| **CDGP** | [Dynamic heterogeneous graph attention neural architecture search](https://ojs.aaai.org/index.php/AAAI/article/view/26338/26110) | Community message passing | Time-aware aggregation | Yes | 2023 |
| **TIGER** | [Tiger: Temporal interaction graph embedding with restarts](https://arxiv.org/pdf/2302.06057) | Implicit (via message passing) | RNN + Dual Memory | Yes | 2023 |
| **RDGL** | [Rdgsl: Dynamic graph representation learning with structure learning](https://arxiv.org/pdf/2309.02025) | Implicit (via message passing) | RNN + Memory | Yes | 2023 |
| **PRES** | [Pres: Toward scalable memory-based dynamic graph neural networks](https://arxiv.org/abs/2402.04284) | Implicit (via message passing) | GMM-guided memory correction | Yes | 2024 |
| **Ada-DyGNN** | [Robust knowledge adaptation for dynamic graph neural networks](https://arxiv.org/pdf/2207.10839) | Reinforced Neighbor Update | Time-based Policy | Yes | 2024 |
| **SEAN** | [Towards adaptive neighborhood for advancing temporal interaction graph modeling](https://arxiv.org/pdf/2406.11891?) | Representative Neighbor Selector | RNN + Temporal-aware aggregation | Yes | 2024 |
| **MemMap** | [Memmap: An adaptive and latent memory structure for dynamic graph learning](https://trace.tennessee.edu/cgi/viewcontent.cgi?article=1006&context=utk_statpubs) | Latent memory-cell grid | Systematic memory routing | Yes | 2024 |
| **MSPipe** | [Mpipe: Efficient temporal gnn training via staleness-aware pipeline](https://arxiv.org/pdf/2402.15113) | Implicit (via message passing) | Staleness-aware update | Yes | 2024 |
|---| **TPP-based Methods** |---|---|---|
| **HTNE** | [Embedding temporal network via neighborhood formation](http://www.shichuan.org/hin/topic/Embedding/2018.KDD%202018%20Embedding%20Temporal%20Network%20via%20Neighborhood%20Formation.pdf) | Historical Neighbor Modeling | Hawkes Process | No | 2018 |
| **M2DNE** | [Temporal network embedding with micro-and macro-dynamics](https://arxiv.org/pdf/1909.04246) | Micro/Macro temporal co-occurrence | Hierarchical TPP | No | 2019 |
| **GHN** | [The graph hawkes network for reasoning on temporal knowledge graphs](https://www.research-collection.ethz.ch/handle/20.500.11850/387992) | Entity-level structure modeling | Hawkes Process | No | 2019 |
| **DyRep** | [Dyrep: Learning representations over dynamic graphs](https://par.nsf.gov/servlets/purl/10099025) | Attentive structural encoding | Multi-scale TPP | No | 2019 |
| **LDG** | [Learning temporal attention in dynamic graphs with bilinear interactions](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0247936&type=printable) | Edge-quality structure adaptive | Adaptive TPP | No | 2021 |
| **TREND** | [Trend: Temporal event and node dynamics for graph representation learning](https://arxiv.org/pdf/2203.14303) | Implicit (via sequential interactions) | Hawkes Process + Transfer function | No | 2022 |
| **DynShare** | [Time-interval aware share recommendation via bi-directional continuous time dynamic graphs](https://scholar.archive.org/work/nu4caelmszgsbemlhkyicmxzyy/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3539618.3591775) | Implicit (via sequential interactions) | Personalized TPP | No | 2023 |
| **EasyDGL** | [Easydsl: Encode, train and interpret for continuous-time dynamic graph learning](https://arxiv.org/pdf/2303.12341) | GAT | TPP + Correlation masking | No | 2024 |
|---| **Random Walk-based Methods** |---|---|---|
| **CTDNE** | [Dynamic network embeddings: From random walks to temporal random walks](https://johnboaz.github.io/files/BigData2018.pdf) | Timestamp-respecting walk | Skip-Gram over walks | No | 2018 |
| **HNIP** | [Temporal network embedding with high-order nonlinear information](https://aaai.org/ojs/index.php/AAAI/article/download/5993/5849) | Temporal random walk | Time-decay in walk sequence | No | 2020 |
| **CAW** | [Inductive representation learning in temporal networks via causal anonymous walks](https://arxiv.org/abs/2101.05974) | Causal Anonymous Walk | Hitting-count encoding | No | 2021 |
| **NeurTWs** | [Neural temporal walks: Motif-aware representation learning on continuous-time dynamic graphs](https://proceedings.neurips.cc/paper_files/paper/2022/file/7dadc855cef7494d5d956a8d28add871-Paper-Conference.pdf) | Motif-guided random walk + ODE | ODE over walk path | No | 2022 |
| **PINT** | [Provably expressive temporal graph networks](https://proceedings.neurips.cc/paper_files/paper/2022/file/d029c97ee0db162c60f2ebc9cb93387e-Paper-Conference.pdf) | Implicit (via message passing) | Provable temporal message passing | No | 2022 |
| **TPNet** | [Improving temporal link prediction via temporal walk matrix projection](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff7bf6014f7826da531aa50f4538ee19-Paper-Conference.pdf) | Time-decayed walk matrix | Temporal relative encoding | No | 2024 |
|---| **Attention/Time Encoding-based Methods** |---|---|---|
| **TGAT** | [Inductive representation learning on temporal graphs](https://arxiv.org/abs/2002.07962) | Temporal self-attention | Functional Time Encoding | No | 2020 |
| **TCL** | [Tcl: Transformer-based dynamic graph modelling via contrastive learning](https://arxiv.org/abs/2105.07944) | Transformer | Functional Time Encoding | No | 2021 |
| **OTGNet** | [Towards open temporal graph neural networks](https://arxiv.org/pdf/2303.15015) | Open graph attention | Extended Time Encoding | No | 2023 |
| **TGRank** | [Expressive and efficient representation learning for ranking links in temporal graphs](https://dl.acm.org/doi/pdf/10.1145/3543507.3583476) | Temporal attention ranking | Enhanced Time Encoding | No | 2023 |
| **DHGAS** | [Community-based dynamic graph learning for popularity prediction](https://dl.acm.org/doi/abs/10.1145/3580305.3599281) | Heterogeneous GNN + Attention | Time Encoding | No | 2023 |
| **DyG2Vec** | [DyG2Vec: Efficient Representation Learning for Dynamic Graphs](https://openreview.net/pdf?id=YRKS2J0x36) | Temporal edge encoding | Time Encoding | No | 2023 |
| **SimpleDyG** | [On the feasibility of simple transformer for dynamic graph modeling](https://dl.acm.org/doi/pdf/10.1145/3589334.3645622) | Transformer | Time and Position Encoding | No | 2024 |
| **Todyformer** | [Todyformer: Towards Holistic Dynamic Graph Transformers with Structure-Aware Tokenization](https://openreview.net/pdf?id=nAQSUqEspb) | GNN+Transformer |  Position Encoding | No | 2024 |
| **DyGFormer** | [Towards better dynamic graph learning: New architecture and unified library](https://proceedings.neurips.cc/paper_files/paper/2023/file/d611019afba70d547bd595e8a4158f55-Paper-Conference.pdf) | 1-hop Neighbour + Co-occurrence | Time and Position Encoding | No | 2024 |
| **APAN** | [Apan: Asynchronous propagation attention network for real-time temporal graph embedding](https://arxiv.org/pdf/2011.11545) | Mailbox + Attention | Time Encoding | Yes | 2021 |
| **iLoRE** | [ilore: Dynamic graph representation with instant long-term modeling and re-occurrence preservation](https://dl.acm.org/doi/pdf/10.1145/3583780.3614926) | Re-occurrence + Identity attention | Time Encoding | Yes | 2022 |
| **TDGNN** | [Continuous-time link prediction via temporal dependent graph neural network](https://dl.acm.org/doi/abs/10.1145/3366423.3380073) | GNN + Time-decay weighting | Exponential Decay Kernel | No | 2020 |
| **DGEL** | [Dynamic graph evolution learning for recommendation](https://dl.acm.org/doi/abs/10.1145/3539618.3591674) | Recent interactions | Time-aware Normalization | No | 2023 |
| **SUPA** | [Instant representation learning for recommendation over large dynamic graphs](https://arxiv.org/pdf/2305.18622) | Implicit (via sequential interactions) | Time modeling mechanisms | No | 2023 |
| **FreeDyG** | [Freedyg: Frequency enhanced continuous-time dynamic graph model for link prediction](https://openreview.net/pdf?id=82Mc5ilInM) | Fourier-enhanced GNN | Functional Time Encoding | No | 2024 |
| **CNE-N** | [Co-neighbor encoding schema: A light-cost structure encoding method for dynamic link prediction](https://arxiv.org/pdf/2407.20871?) | Hash table-based memory | Temporal-diverse memory | Yes | 2024 |
| **TG-Mixer** | [Interactions exhibit clustering rhythm: A prevalent observation for advancing temporal link prediction](https://arxiv.org/abs/2308.14129) | Clustering Patterns | Time Encoding | No | 2024 |
| **DyGMamba** | [DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models](https://openreview.net/pdf?id=sq5AJvVuha) | Mamba | Time Encoding | No | 2025 |
|---| **MLP-based Methods** |---|---|---|
| **GraphMixer** | [Do we really need complicated model architectures for temporal networks?](https://arxiv.org/pdf/2302.11636) | MLP + Mean pooling | Fixed Time Encoding | No | 2024 |
| **RepeatMixer** | [Repeat-aware neighbor sampling for dynamic graph learning](https://arxiv.org/pdf/2405.17473?) | MLP + Repeat-aware sampling | Time-aware aggregation | No | 2024 |
| **BandRank** | [Ranking on dynamic graphs: An effective and robust band-pass disentangled approach](https://openreview.net/pdf?id=cah0ZYeMz0) | Frequency-band MLP | Band-pass Time Filters | No | 2025 |

### Benchmarks
| Benchmark    | Paper                                                                                                                                              | Repository                                                                  | Specialize       | Year |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|------------------|:----:|
| PyG-Temporal | [arXiv](https://arxiv.org/pdf/2104.07788)                                                                                                          | [GitHub](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) | DTDG             | 2021 |
| TGL          | [arXiv](https://arxiv.org/abs/2203.14883)                                                                                                          | [GitHub](https://github.com/amazon-science/tgl)                             | Large-scale CTDG | 2022 |
| SPEED        | [arXiv](https://arxiv.org/abs/2308.14129)                                                                                                          | [GitHub](https://github.com/chenxi1228/SPEED)                               | Large-scale CTDG | 2023 |
| DYGL         | [APWeb](https://link.springer.com/chapter/10.1007/978-981-97-2387-4_26)                                                                            | [GitHub](https://github.com/half-salve/DYGL-lib)                            | DTDG and CTDG    | 2023 |
| DyGLib       | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d611019afba70d547bd595e8a4158f55-Abstract-Conference.html)                    | [GitHub](https://github.com/yule-BUAA/DyGLib)                               | CTDG             | 2024 |
| TGB          | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fda026cf2423a01fcbcf1e1e43ee9a50-Abstract-Datasets_and_Benchmarks_Track.html) | [GitHub](https://github.com/shenyangHuang/TGB)                              | CTDG             | 2024 |
| BenchTemp    | [ICDE](https://www.computer.org/csdl/proceedings-article/icde/2024/1.715E49/1YOu0hGqfW8)                                                           | [GitHub](https://github.com/johnnyhuangcs/benchtemp)                        | CTDG             | 2024 |
| DGB          | [TNNLS](https://ieeexplore.ieee.org/abstract/document/10490120)                                                                                    | [GitHub](https://github.com/gravins/dynamic_graph_benchmark)                | DTDG and CTDG    | 2024 |
| BenchTGNN    | [arXiv](https://arxiv.org/abs/2412.20256)                                                                                                          | [GitHub](https://github.com/Yang-yuxin/BenchTGNN)                           | CTDG             | 2024 |
| TGX          | [WSDM](https://dl.acm.org/doi/abs/10.1145/3616855.3635694)                                                                                         | [GitHub](https://github.com/ComplexData-MILA/TGX)                           | CTDG             | 2024 |
| DGNN         | [arXiv](https://arxiv.org/abs/2405.00476)                                                                                                          | [GitHub](https://github.com/fengwudi/DGNN_model_and_data)                   | DTDG and CTDG    | 2024 |
| UTG          | [LoG](https://openreview.net/forum?id=ZKHV6Cpsxg)                                                                                                  | [GitHub](https://github.com/shenyangHuang/UTG)                              | DTDG and CTDG    | 2024 |

### Dataset Resources

[SNAP](https://snap.stanford.edu/data/index.html)

[SPEED Datasets](https://github.com/chenxi1228/SPEED)

[TGB Datasets](https://tgb.complexdatalab.com/docs/dataset_overview/)

[DyGLib Datasets](https://zenodo.org/records/7213796#.Y1cO6y8r30o)




## Textual Data
### Papers
| Methods | Paper Title | LLM Function | Modality Transformation | LLM Fine-tuning | Year |
|---|---|---|---|---|---|
|---| **LLMs for Augmenting** |---|---|---|
| **Carranza et al.** | [Privacy-preserving recommender systems with synthetic query generation using differentially private large language models](https://arxiv.org/abs/2305.05973) | Behavior Augmentation | Language | No | 2023 |
| **TF-DCon** | [Leveraging large language models (llms) to empower training-free dataset condensation for content-based recommendation](https://arxiv.org/abs/2310.09874) | Behavior Augmentation | Language | No | 2023 |
| **CUP** | [Recommendations by concise user profiles from review text](https://arxiv.org/abs/2311.01314) | Feature Augmentation | Language | No | 2023 |
| **Precious2GPT** | [Precious2gpt: the combination of multicomics pretrained transformer and conditional diffusion for artificial multi-omics multi-species multi-tissue sample generation](https://www.nature.com/articles/s41514-024-00163-3.pdf) | Behavior Augmentation | Multi-omics → Language | Yes | 2024 |
| **TriSum** | [Trisum: Learning summarization ability from large language models with structured rationale](https://arxiv.org/abs/2403.10351) | Behavior Augmentation | Sequence → Language | Yes | 2024 |
| **Chen et al.** | [Beyond numbers: Creating analogies to enhance data comprehension and communication with generative ai](https://dl.acm.org/doi/abs/10.1145/3613904.3642480) | Behavior Augmentation | Sequence → Language | No | 2024 |
| **KAR** | [Towards open-world recommendation with knowledge augmentation from large language models](https://arxiv.org/pdf/2306.10933) | Feature Augmentation | Language | No | 2024 |
| **Ghanem et al.** | [Fine-tuning vs. prompting: evaluating the knowledge graph construction with llms](https://hal.science/hal-04862235v1/file/221.pdf) | Feature Augmentation | Graph → Language | Yes | 2024 |
| **Ghanem et al.** | [Enhancing knowledge graph construction: Evaluating with emphasis on hallucination, omission, and graph similarity metrics](https://arxiv.org/pdf/2502.05239) | Feature Augmentation | Graph → Language | No | 2024 |
| **APLe** | [Aple: Tokenwise adaptive for multi-modal prompt learning](https://arxiv.org/abs/2401.06827) | Feature Augmentation | Sequence (multi-modal) → Language | Yes | 2024 |
| **KAG** | [Kag: Boosting llms in professional domains via knowledge augmented generation](https://dl.acm.org/doi/pdf/10.1145/3701716.3715240) | Behavior Augmentation | Language | Yes | 2025 |
| **SAGCN** | [Understanding before recommendation: Semantic aspect-aware review exploitation via large language models](https://dl.acm.org/doi/pdf/10.1145/3704999) | Feature Augmentation | Sequence → Language | No | 2025 |
| **Pandey et al.** | [Generating product reviews from aspect-based ratings using large language models](https://www.sciencedirect.com/science/article/pii/S0969698925000232) | Feature Augmentation | Sequence → Language | Yes | 2025 |
| **DyLas** | [Dylas: A dynamic label alignment strategy for large-scale multi-label text classification](https://www.sciencedirect.com/science/article/pii/S156625352500154X) | Feature Augmentation | Language | Yes | 2025 |
|---| **LLMs for Encoding** |---|---|---|
| **U-BERT** | [U-bert: Pre-training user representations for improved recommendation](https://ojs.aaai.org/index.php/AAAI/article/download/16557/16364) | Representation Enhancement | Language | Yes | 2021 |
| **Social-LLM** | [Social-llm: Modeling user behavior at scale using language models and social network data](https://arxiv.org/abs/2401.00893) | Representation Enhancement | Graph → Language | No | 2023 |
| **Brooks et al.** | [Emotion expression estimates to measure and improve multimodal social-affective interactions](https://dl.acm.org/doi/abs/10.1145/3610661.3616129) | Cross-modality Unification | Sequence → Language | No | 2023 |
| **UFIN** | [Ufin: Universal feature interaction network for multi-domain click-through rate prediction](https://arxiv.org/abs/2311.15493) | Cross-modality Unification | Sequence → Language | No | 2023 |
| **Uni_CTR** | [A unified framework for multi-domain ctr prediction via large language models](https://arxiv.org/pdf/2312.10743) | Cross-modality Unification | Sequence → Language | No | 2023 |
| **GNR** | [Generative news recommendation](https://dl.acm.org/doi/pdf/10.1145/3589334.3645448) | Representation Enhancement | Language | Yes | 2024 |
| **EAGER** | [Eager: Two-stream generative recommender with behavior-semantic collaboration](https://arxiv.org/pdf/2406.14017) | Representation Enhancement | Sequence → Language | Yes | 2024 |
| **LC-Re** | [Adapting large language models by integrating collaborative semantics for recommendation](https://arxiv.org/pdf/2311.09049) | Representation Enhancement | Language | Yes | 2024 |
| **OneLLM** | [Onellm: One framework to align all modalities with language](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_OneLLM_One_Framework_to_Align_All_Modalities_with_Language_CVPR_2024_paper.pdf) | Cross-modality Unification | Graph/Sequence → Language | Yes | 2024 |
| **LC-Re** | [Adapting large language models by integrating collaborative semantics for recommendation](https://arxiv.org/pdf/2311.09049) | Cross-modality Unification | Language | Yes | 2024 |
| **Lu et al.** | [A large language model-based approach for personalized search results re-ranking in professional domains](http://academianexusjournal.com/index.php/anj/article/download/5/6) | Representation Enhancement | Language | Yes | 2025 |
| **Chavinda et al.** | [A dual contrastive learning framework for enhanced hate speech detection in low-resource languages](https://aclanthology.org/2025.chipsal-1.11.pdf) | Representation Enhancement | Language | Yes | 2025 |
| **Poison-RAG** | [Poison-rag: Adversarial data poisoning attacks on retrieval-augmented generation in recommender systems](https://arxiv.org/pdf/2501.11759) | Representation Enhancement | Sequence → Language | No | 2025 |
| **RUNSRec** | [Enhanced universal sequence representation learning for recommender systems](https://dl.acm.org/doi/abs/10.1145/3717832) | Cross-modality Unification | Sequence → Language | Yes | 2025 |
| **Socialmind** | [Socialmind: LIm-based proactive ar social assistive system with human-like perception for in-situ live interactions](https://dl.acm.org/doi/pdf/10.1145/3712286) | Cross-modality Unification | Graph/Sequence → Language | No | 2025 |
| **CROSS** | [Unifying text semantics and graph structures for temporal text-attributed graphs with large language models](https://arxiv.org/abs/2503.14411) | Cross-modality Unification | Graph → Language | Yes | 2025 |
| **Lin et al.** | [Large language models make sample-efficient recommender systems](https://arxiv.org/pdf/2406.02368) | Cross-modality Unification | Sequence → Language | No | 2025 |
|---| **LLMs for Controlling** |---|---|---|
| **RecLLM** | [Leveraging large language models in conversational recommender systems](https://arxiv.org/abs/2305.07961) | Pipeline Control | Language | No | 2023 |
| **RecMind** | [Recmind: Large language model powered agent for recommendation](https://arxiv.org/abs/2308.14296) | Pipeline Control | Language | No | 2023 |
| **FinCon** | [Fincon: A synthesized llm multi-agent system with conceptual verbal reinforcement for enhanced financial decision making](https://proceedings.neurips.cc/paper_files/paper/2024/file/f7ae4fe91d96f50abc2211f09b6a7e49-Paper-Conference.pdf) | Pipeline Control | Language | Yes | 2024 |
| **FinAgent** | [A multimodal foundation agent for financial trading: Tool-augmented, diversified, and generalist](https://dl.acm.org/doi/pdf/10.1145/3637528.3671801) | Pipeline Control | Language | Yes | 2024 |
| **TradingAgents** | [Tradingagents: Multiagents llm financial trading framework](https://arxiv.org/pdf/2412.20138?) | Pipeline Control | Language | Yes | 2025 |
| **TS-Reasoner** | [Domain-oriented time series inference agents for reasoning and automated analysis](https://arxiv.org/pdf/2410.04047) | Pipeline Control | Language | Yes | 2025 |
| **RiskLabs** | [Risklabs: Predicting financial risk using large language model based on multimodal and multi-sources data](https://ideas.repec.org/p/arx/papers/2404.07452.html) | Pipeline Control | Language | Yes | 2025 |
| **AgentSociety** | [Agentsociety: Large-scale simulation of llm-driven generative agents advances understanding of human behaviors and society](https://arxiv.org/abs/2502.08691) | Pipeline Control | Language | Yes | 2025 |
| **ProSim** | [Simulating prosocial behavior and social contagion in llm agents under institutional interventions](https://arxiv.org/pdf/2505.15857) | Pipeline Control | Language | Yes | 2025 |

### Benchmarks

| Benchmark    | Paper                                           |Repository                                | Year |
|--------------|-------------------------------------------------|------------------------------------------|:----:|
| DTGB         |[NeurIPS](https://arxiv.org/abs/2406.12072)      | [GitHub](https://github.com/zjs123/DTGB) | 2024 |


### Dataset Resources

[DTGB-Google Drive](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk)


## Application Scenarios

| Method                                                   | Paper Title | Application Scenarios | Sub-Scenarios |
|----------------------------------------------------------|---|---|---|
| **NCF**                                                  | [Neural collaborative filtering](<https://arxiv.org/pdf/1708.05031>) | E-commerce | Personalized Search and Recommendation |
| **DeepFM**                                               | [Deepfm: a factorization-machine based neural network for ctr prediction](<https://arxiv.org/abs/1703.04247>) | E-commerce | Personalized Search and Recommendation |
| **xDeepFM**                                              | [xdeepfm: Combining explicit and implicit feature interactions for recommender systems](<https://arxiv.org/pdf/1803.05170>) | E-commerce | Personalized Search and Recommendation |
| **AutoInt**                                              | [Autoint: Automatic feature interaction learning via self-attentive neural networks](<https://arxiv.org/pdf/1810.11921>) | E-commerce | Personalized Search and Recommendation |
| **FPMC**                                                 | [Factorizing personalized markov chains for next-basket recommendation](<http://www.ambuehler.ethz.ch/CDstore/www2010/www/p811.pdf>) | E-commerce | Personalized Search and Recommendation |
| **GRU4Rec**                                              | [Session-based recommendations with recurrent neural networks](<https://arxiv.org/abs/1511.06939>) | E-commerce | Personalized Search and Recommendation |
| **SAS-Rec**                                              | [Self-attentive sequential recommendation](<https://peerj.com/articles/cs-1701.pdf>) | E-commerce | Personalized Search and Recommendation |
| **MARN**                                                 | [Adversarial multimodal representation learning for click-through rate prediction](<https://arxiv.org/pdf/2003.07162>) | E-commerce | Personalized Search and Recommendation |
| **SAGE**                                                 | [Sequence as genes: an user behavior modeling framework for fraud transaction detection in e-commerce](<https://dl.acm.org/doi/abs/10.1145/3580305.3599905>) | E-commerce | Personalized Search and Recommendation |
| **IU4Rec**                                               | [U4rec: Interest unit-based product organization and recommendation for e-commerce platform](<https://arxiv.org/pdf/2502.07658>) | E-commerce | Personalized Search and Recommendation |
| **DGSR**                                                 | [Dynamic graph neural networks for sequential recommendation](<https://arxiv.org/pdf/2104.07368>) | E-commerce | Personalized Search and Recommendation |
| **EAGLE**                                                | [When speed meets accuracy: an efficient and effective graph model for temporal link prediction](<https://arxiv.org/abs/2507.13825>) | E-commerce | Personalized Search and Recommendation |
| **SRJGraph**                                             | [Joint learning of e-commerce search and recommendation with a unified graph neural network](<https://www.researchgate.net/profile/Tao-Zhuang-4/publication/358630042_Joint_Learning_of_E-commerce_Search_and_Recommendation_with_a_Unified_Graph_Neural_Network/links/626bb348ef19bf606decfb77/Joint-Learning-of-E-commerce-Search-and-Recommendation-with-a-Unified-Graph-Neural-Network.pdf>) | E-commerce | Personalized Search and Recommendation |
| **BERT4Rec**                                             | [Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer](<https://arxiv.org/pdf/1904.06690>) | E-commerce | Personalized Search and Recommendation |
| **P5**                                                   | [Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)](<https://arxiv.org/pdf/2203.13366>) | E-commerce | Personalized Search and Recommendation |
| **LLM4Rec**                                              | [A survey on large language models for recommendation](<https://arxiv.org/pdf/2305.19860>) | E-commerce | Personalized Search and Recommendation |
| **Chat-REC**                                             | [Chatrec: Towards interactive and explainable llms-augmented recommender system](<https://arxiv.org/pdf/2303.14524>) | E-commerce | Personalized Search and Recommendation |
| **M6-Rec**                                               | [M6-rec: Generative pretrained language models are open-ended recommender systems](<https://arxiv.org/pdf/2205.08084>) | E-commerce | Personalized Search and Recommendation |
| **V-RECS**                                               | [V-recs, a low-cost llm4vis recommender with explanations, captioning and suggestions](<https://arxiv.org/pdf/2406.15259?>) | E-commerce | Personalized Search and Recommendation |
| **FTRL-Proximal**                                        | [Ad click prediction: a view from the trenches](<https://dl.acm.org/doi/pdf/10.1145/2487575.2488200>) | E-commerce | Enhanced User Understanding and Profiling |
| **RBP**                                                  | [Repeat buyer prediction for e-commerce](<https://www.kdd.org/kdd2016/papers/files/adf0160-liuA.pdf>) | E-commerce | Enhanced User Understanding and Profiling |
| **DUPN**                                                 | [Perceive your users in depth: Learning universal user representations from multiple e-commerce tasks](<https://arxiv.org/pdf/1805.10727>) | E-commerce | Enhanced User Understanding and Profiling |
| **HUP**                                                  | [Hierarchical user profiling for e-commerce recommender systems](<https://guyulongcs.github.io/files/WSDM2020_HUP.pdf>) | E-commerce | Enhanced User Understanding and Profiling |
| **TriPer**                                               | [Persona identification in e-commerce with scarce labels and in-context graph learning](<https://dl.acm.org/doi/pdf/10.1145/3711896.3737080>) | E-commerce | Enhanced User Understanding and Profiling |
| **GPG**                                                  | [Guided profile generation improves personalization with large language models](<https://arxiv.org/pdf/2409.13093>) | E-commerce | Enhanced User Understanding and Profiling |
| **Sabouri et. al**                                       | [Towards explainable temporal user profiling with llms](<https://arxiv.org/pdf/2505.00886>) | E-commerce | Enhanced User Understanding and Profiling |
| **PURE**                                                 | [Llm-based user profile management for recommender system](<https://arxiv.org/pdf/2502.14541?>) | E-commerce | Enhanced User Understanding and Profiling |
| **RETAIN**                                               | [Retain: An interpretable predictive model for healthcare using reverse time attention mechanism](<https://proceedings.neurips.cc/paper/2016/file/231141b34c82aa95e48810a9d1b33a79-Paper.pdf>) | Healthcare | Disease Prediction and Risk Stratification |
| **BEHRT**                                                | [Behrt: transformer for electronic health records](<https://www.nature.com/articles/s41598-020-62922-y.pdf>) | Healthcare | Disease Prediction and Risk Stratification |
| **ProtoEHR**                                             | [Protoehr: Hierarchical prototype learning for ehr-based healthcare predictions](<https://arxiv.org/abs/2508.18313>) | Healthcare | Disease Prediction and Risk Stratification |
| **MHGRL**                                                | [Mhgrl: An effective representation learning model for electronic health records](<https://aclanthology.org/2024.lrec-main.985.pdf>) | Healthcare | Disease Prediction and Risk Stratification |
| **EHRAgent**                                             | [Ehangent: Code empowers large language models for few-shot complex tabular reasoning on electronic health records](<https://aclanthology.org/2024.emnlp-main.1245.pdf>) | Healthcare | Disease Prediction and Risk Stratification |
| **DeepSurv**                                             | [Deepsurv: personalized treatment recommender system using a cox proportional hazards deep neural network](<https://link.springer.com/content/pdf/10.1186/s12874-018-0482-1.pdf>) | Healthcare | Personalized Treatment Recommendation |
| **Lee and Hauskrecht**                                   | [Personalized event prediction for electronic health records](<https://www.sciencedirect.com/science/article/am/pii/S0933365723001343>) | Healthcare | Personalized Treatment Recommendation |
| **SRL-RNN**                                              | [Supervised reinforcement learning with recurrent neural network for dynamic treatment recommendation](<https://arxiv.org/pdf/1807.01473>) | Healthcare | Personalized Treatment Recommendation |
| **KARE**                                                 | [Reasoning-enhanced healthcare predictions with knowledge graph community retrieval](<https://arxiv.org/abs/2410.04585>) | Healthcare | Personalized Treatment Recommendation |
| **GAME**                                                 | [Representation learning to advance multi-institutional studies with electronic health record data](<https://arxiv.org/abs/2502.08547>) | Healthcare | Personalized Treatment Recommendation |
| **Med-PaLM 2**                                           | [Toward expert-level medical question answering with large language models](<https://www.nature.com/articles/s41591-024-03423-7.pdf>) | Healthcare | Personalized Treatment Recommendation |
| **Peters and Matz**    | [Large language models can infer psychological dispositions of social media users](<https://academic.oup.com/pnasnexus/article-pdf/3/6/pgae231/58356613/pgae231.pdf>) | Social Media | User Profiling and Interest Modeling |
| **LLM-TUP**    | [Towards explainable temporal user profiling with llms](<https://arxiv.org/pdf/2505.00886>) | Social Media | User Profiling and Interest Modeling |
| **GREENER**                                              | [Greener: Graph neural networks for news media profiling](<https://arxiv.org/pdf/2211.05533>) | Social Media | User Profiling and Interest Modeling |
| **HL-MRFs**                                              | [User profiling using hinge-loss markov random fields](<https://arxiv.org/pdf/2001.01177>) | Social Media | User Profiling and Interest Modeling |
| **SA-LSTM**                                           | [Social-aware sequential modeling of user interests: A deep learning approach](<https://ieeexplore.ieee.org/abstract/document/8486686/>) | Social Media | User Profiling and Interest Modeling |
| **UPO**                                          | [Temporal user interest modeling for online advertising using bi-lstm network improved by an updated version of parrot optimizer](<https://www.nature.com/articles/s41598-025-03208-z.pdf>) | Social Media | User Profiling and Interest Modeling |
| **SPAR**                                                 | [Spar: Personalized content-based recommendation via long engagement attention](<https://arxiv.org/pdf/2402.10555>) | Social Media | Content Recommendation and Feed Ranking |
| **LRD**                                           | [Sequential recommendation with latent relations based on large language model](<https://dl.acm.org/doi/pdf/10.1145/3626772.3657762>) | Social Media | Content Recommendation and Feed Ranking |
| **Mamba4Rec**                                            | [Mamba4rec: Towards efficient sequential recommendation with selective state space models](<https://arxiv.org/pdf/2403.03900>) | Social Media | Content Recommendation and Feed Ranking |
| **ContextGNN**                                           | [Contextgnn: Beyond two-tower recommendation systems](<https://arxiv.org/pdf/2411.19513>) | Social Media | Content Recommendation and Feed Ranking |
| **CGAT**                                                 | [Contextualized graph attention network for recommendation with item knowledge graph](<https://dr.ntu.edu.sg/bitstream/10356/156036/2/Contextualized_Graph_Attention_Network_for_Recommendation_with_Item_Knowledge_Graph.pdf>) | Social Media | Content Recommendation and Feed Ranking |
| **Chen et. al**                                          | [Explore the potential of llms in misinformation detection: An empirical study](<https://openreview.net/pdf?id=W2zawmik8i>) | Social Media | Misinformation and Rumor Detection |
| **Ruiz et. al**                                          | [An explainable framework for misinformation identification via critical question answering](<https://arxiv.org/pdf/2503.14626>) | Social Media | Misinformation and Rumor Detection |
| **Liu et. al**                                           | [Rumor detection with a novel graph neural network approach](<https://arxiv.org/pdf/2403.16206>) | Social Media | Misinformation and Rumor Detection |
| **GSMA**                                                 | [Rumor detection model with weighted graphsage focusing on node location](<https://www.nature.com/articles/s41598-024-76738-7.pdf>) | Social Media | Misinformation and Rumor Detection |
| **Dong et. al**                                          | [Semi-supervised bidirectional rnn for misinformation detection](<https://www.sciencedirect.com/science/article/am/pii/S2666827022001037>) | Social Media | Misinformation and Rumor Detection |
| **GETAE**                                                | [Getae: Graph information enhanced deep neural network ensemble architecture for fake news detection](<https://www.sciencedirect.com/science/article/pii/S0957417425006062>) | Social Media | Misinformation and Rumor Detection |
| **MAGIC**                                                | [A multimodal adaptive graph-based intelligent classification model for fake news](<https://arxiv.org/pdf/2411.06097>) | Social Media | Misinformation and Rumor Detection |
| **Tiny-toxic-detector**                                  | [Tiny-toxic-detector: A compact transformer-based model for toxic content detection](<https://arxiv.org/pdf/2409.02114?>) | Social Media | Toxicity and Safety Moderation |
| **MetaTox**                                              | [Enhancing llm-based hatred and toxicity detection with meta-toxic knowledge graph](<https://aclanthology.org/2025.findings-acl.1270.pdf>) | Social Media | Toxicity and Safety Moderation |
| **ShieldVLM**                                            | [Shieldvlm: Safeguarding the multimodal implicit toxicity via deliberative reasoning with lvlms](<https://arxiv.org/pdf/2505.14035>) | Social Media | Toxicity and Safety Moderation |
| **Cheng et. al**                                         | [Bias mitigation for toxicity detection via sequential decisions](<https://dl.acm.org/doi/pdf/10.1145/3477495.3531945>) | Social Media | Toxicity and Safety Moderation |
| **Oyetayo et. al**                                             | [A neural multi-label model for enhanced toxicity detection in online conversations](<https://aujs.adelekeuniversity.edu.ng/index.php/aujs/article/download/151/91>) | Social Media | Toxicity and Safety Moderation |
| **MHSDF**                                        | [A comprehensive framework for multi-modal hate speech detection in social media using deep learning](<https://www.nature.com/articles/s41598-025-94069-z.pdf>) | Social Media | Toxicity and Safety Moderation |
| **Xavier et al.**                                        | [Towards the identification of money laundering in bank transactions using recurrent neural networks](<https://ieeexplore.ieee.org/abstract/document/10814854>) | Social Media | Toxicity and Safety Moderation |
| **Branco et al.**                       | [Interleaved sequence rnns for fraud detection](<https://dl.acm.org/doi/pdf/10.1145/3394486.3403361>) | Finance | Risk Management and Fraud Detection |
| **FinDeepBehaviorCluster** | [Explainable deep behavioral sequence clustering for transaction fraud detection](<https://arxiv.org/abs/2101.04285>) | Finance | Risk Management and Fraud Detection |
| **UB-PTM**                   | [User behavior pre-training for online fraud detection](<https://dl.acm.org/doi/pdf/10.1145/3534678.3539126>) | Finance | Risk Management and Fraud Detection |
| **CLeAR**                | [Robust sequence-based self-supervised representation learning for anti-money laundering](<https://dl.acm.org/doi/abs/10.1145/3627673.3680078>) | Finance | Risk Management and Fraud Detection |
| **DyHDGE**                      | [Dyhedge: Dynamic heterogeneous transaction graph embedding for safety-centric fraud detection in financial scenarios](<https://www.sciencedirect.com/science/article/pii/S2666449624000483>) | Finance | Risk Management and Fraud Detection |
| **EvolveGCN**                                            | [Evolvegcn: Evolving graph convolutional networks for dynamic graphs](<https://aaai.org/ojs/index.php/AAAI/article/view/5984/5840>) | Finance | Risk Management and Fraud Detection |
| **FiFrauD**              | [Fifraud: unsupervised financial fraud detection in dynamic graph streams](<https://dl.acm.org/doi/abs/10.1145/3641857>) | Finance | Risk Management and Fraud Detection |
| **GADY**                     | [Gady: Unsupervised anomaly detection on dynamic graphs](<https://arxiv.org/abs/2310.16376>) | Finance | Risk Management and Fraud Detection |
| **EL$^2$-DGAD**                                                 | [Semisupervised anomaly detection with extremely limited labels in dynamic graphs](<https://arxiv.org/abs/2501.15035>) | Finance | Risk Management and Fraud Detection |
| **SSH-T$^3$**                                                 | [SSH-T$^3$: A Hierarchical Pre-training Framework for Multi-Scenario Financial Risk Assessment](<https://dl.acm.org/doi/abs/10.1145/3746252.3761504>) | Finance | Risk Management and Fraud Detection |
| **Jiang**                | [Detecting scams using large language models](<https://arxiv.org/pdf/2402.03147>) | Finance | Risk Management and Fraud Detection |
| **Korkanti**                  | [Enhancing financial fraud detection using llms and advanced data analytics](<https://ieeexplore.ieee.org/abstract/document/10760895/>) | Finance | Risk Management and Fraud Detection |
| **Singh et al.**                     | [Advanced real-time fraud detection using rag-based llms](<https://arxiv.org/pdf/2501.15290?>) | Finance | Risk Management and Fraud Detection |
| **RiskLabs**             | [Risklabs: Predicting financial risk using large language model based on multimodal and multi-sources data](<https://ideas.repec.org/p/arx/papers/2404.07452.html>) | Finance | Risk Management and Fraud Detection |
| **Ta et al.**               | [Portfolio optimization-based stock prediction using long-short term memory network in quantitative trading](<https://www.mdpi.com/2076-3417/10/2/437>) | Finance | Stock Forecasting and Portfolio Optimization |
| **Wang et al.**               | [A deep learning integrated framework for predicting stock index price and fluctuation via singular spectrum analysis and particle swarm optimization](<https://dl.acm.org/doi/abs/10.1007/s10489-024-05271-x>) | Finance | Stock Forecasting and Portfolio Optimization |
| **Zhang et al.**               | [Prediction model of stock return on investment based on hybrid dnn and tabnet model](<https://peerj.com/articles/cs-2057/>) | Finance | Stock Forecasting and Portfolio Optimization |
| **Lezmi and Xu**          | [Time series forecasting with transformer models and application to asset management](<https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4375798>) | Finance | Stock Forecasting and Portfolio Optimization |
| **SARL**                        | [Reinforcement-learning based portfolio management with augmented asset movement prediction states](<https://ojs.aaai.org/index.php/AAAI/article/view/5462/5318>) | Finance | Stock Forecasting and Portfolio Optimization |
| **StockTime**                                        | [Stocktime: A time series specialized large language model architecture for stock price prediction](<https://arxiv.org/pdf/2409.08281?>) | Finance | Stock Forecasting and Portfolio Optimization |
| **LED-GNN**                                        | [Modeling interactions between stocks using LLM-enhanced graphs for volume prediction](<https://aclanthology.org/2025.finnlp-1.14.pdf>) | Finance | Stock Forecasting and Portfolio Optimization |
| **SEP**            | [Learning to generate explainable stock predictions using self-reflective large language models](<https://dl.acm.org/doi/pdf/10.1145/3589334.3645611>) | Finance | Stock Forecasting and Portfolio Optimization |
| **Abe et al.**                  | [Leveraging large language models for institutional portfolio management: Persona-based ensembles](<https://arxiv.org/pdf/2411.19515>) | Finance | Stock Forecasting and Portfolio Optimization |
| **Min et al.**       | [Player goal recognition in open-world digital games with long short-term memory networks](<https://www.ijcai.org/Proceedings/16/Papers/368.pdf>) | Gaming | Player Goal and Strategy Recognition |
| **Kantharaju and Ontañón**          | [Discovering meaningful labelings for rts game replays via replay embeddings](<https://ieee-cog.org/2020/papers/paper_60.pdf>) | Gaming | Player Goal and Strategy Recognition |
| **player2vec**        | [player2vec: A language modeling approach to understand player behavior in games](<https://arxiv.org/pdf/2404.04234>) | Gaming | Player Goal and Strategy Recognition |
| **MTC**            | [Understanding players as if they are talking to the game in a customized language: A pilot study](<https://aclanthology.org/2024.customnlp4u-1.5.pdf>) | Gaming | Player Goal and Strategy Recognition |
| **Zhao et al.**                 | [Player behavior modeling for enhancing role-playing game engagement](<http://www.shazhao.net/papers/tcss.pdf>) | Gaming | NPC Generation and Control |
| **MineDojo**                     | [Minedojo: Building open-ended embodied agents with internet-scale knowledge](<https://proceedings.neurips.cc/paper_files/paper/2022/file/74a67268c5cc5910f64938cac4526a90-Paper-Datasets_and_Benchmarks.pdf>) | Gaming | NPC Generation and Control |
| **Voyager**  | [Voyager: An open-ended embodied agent with large language models](<https://arxiv.org/pdf/2305.16291>) | Gaming | NPC Generation and Control |
| **Yuan et al.**                 | [Skill reinforcement learning and planning for open-world long-horizon tasks](<https://arxiv.org/pdf/2303.16563>) | Gaming | NPC Generation and Control |
| **Volum et al.**              | [Craft an iron sword: Dynamically generating interactive game characters by prompting large language models tuned on code](<https://aclanthology.org/2022.wordplay-1.3.pdf>) | Gaming | NPC Generation and Control |
| **Kwak et al.**              | [Linguistic analysis of toxic behavior in an online video game](<https://arxiv.org/pdf/1410.5185>) | Gaming | Toxicity Detection and Community Management |
| **Aguerri et al.**               | [The enemy hates best? toxicity in league of legends and its content moderation implications](<https://link.springer.com/content/pdf/10.1007/s10610-023-09541-1.pdf>) | Gaming | Toxicity Detection and Community Management |
| **Märtens**                   | [Toxicity detection in multiplayer online games](<https://ieeexplore.ieee.org/abstract/document/7382991>) | Gaming | Toxicity Detection and Community Management |
| **Ng et al.**            | [Challenges for real-time toxicity detection in online games](<https://arxiv.org/pdf/2407.04383>) | Gaming | Toxicity Detection and Community Management |
| **Morrier et al.**               | [Reinforcement learning for efficient toxicity detection in competitive online video games](<https://arxiv.org/pdf/2503.20968?>) | Gaming | Toxicity Detection and Community Management |


## Key Modules

| Method | Paper Title | Module |
|---|---|---|
| **FTR** | [Time2vec: Learning a vector representation of time](<https://arxiv.org/abs/1907.05321>) | Time Encoding | 
| **FTR** | [Self-attention with functional time representation learning](<https://proceedings.neurips.cc/paper/2019/file/cf34645d98a7630e2bcca98b3e29c8f2-Paper.pdf>) | Time Encoding |
| **Time2Vec** | [Time2vec: Learning a vector representation of time](<https://arxiv.org/abs/1907.05321>) | Time Encoding | 
| **LeTE** | [Rethinking time encoding via learnable transformation functions](<https://proceedings.mlr.press/v267/chen25bh.html>) | Time Encoding | 
| **Linear Time Encoding** | [Between Linear and Sinusoidal: Rethinking the Time Encoder in Dynamic Graph Learning](<https://openreview.net/pdf?id=W6GQvdOGHg>) | Time Encoding | 
| **Absolute PE** | [Attention is all you need](<https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>) | Positional Encoding |
| **RoPE** | [Rofomer: Enhanced transformer with rotary position embedding](<https://arxiv.org/pdf/2104.09864>) | Positional Encoding | 







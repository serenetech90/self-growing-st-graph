# self-growing-st-graph
Cite as: @inproceedings{haddad2020self,
  title={Self-Growing Spatial Graph Networks for Pedestrian Trajectory Prediction},
  author={Haddad, Sirin and Lam, Siew-Kei},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1151--1159},
  year={2020}
}

Typo correction: Eq18 in the main paper, states the calculation for ADE metric, the old version doesn't match py code:
correct version:
E = \frac{\sum_{i = 1}^N \sqrt{\sum_{j = 1}^l (\widetilde{X_i^j} - X_i^j)^2 } }{N * l} 

Old version: 
E = \frac{\sqrt{\sum_{i = 1}^N \sum_{j = 1}^l (\widetilde{X_i^j} - X_i^j)^2 } }{N * l} 

**Changelog (Documentation Updates)**

* Added section explaining **why India was chosen as the geographic focus** for the dataset and model evaluation.
* Added justification for using the **RDD2022 dataset** and its relevance for road damage detection tasks.
* Included **citation to the original RDD2022 research paper and dataset source**.
* Added a **Data Governance** section covering dataset sourcing, usage permissions, and responsible data handling.
* Documented **how the road images were captured** (camera setup, vehicle-mounted capture process, and typical conditions).
* Added explanation of the **XML annotation file format** used in the dataset.
* Included a detailed description of **bounding box structure within the XML annotations** (object tags, coordinates, class labels).
* Embedded **sample images from the Jupyter notebook (IPYNB)** into the documentation file for visual reference.
* Added description of the **train–test split strategy** used in the experiments.
* Listed and documented **all dataset classes** used in the project.
* Added a **separate section addressing class imbalance**, including:

  * Analysis of imbalance across damage classes
  * Explanation of how **YOLO architecture handles class imbalance during training**.
* Implemented and documented **stratified dataset splitting** to maintain class distribution between splits.
* Added a dedicated section explaining **how data leakage is prevented** during dataset preparation and splitting.
* Included analysis of **extremely large and extremely small bounding boxes**, with counts and handling strategy.
* Added **duplicate image detection checks** and methodology used to identify/remove duplicates.
* Clarified whether preprocessing involves **only image resizing or also data augmentation**.
* Documented the **data augmentation techniques and algorithms used**.
* Added explanation of **how image resizing affects bounding boxes and model training**.
* Added a **separate detailed section on data leakage prevention**, including dataset isolation and preprocessing safeguards.

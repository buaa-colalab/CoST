<div align="center">

<h1>CoST: Efficient Collaborative Perception From Unified Spatiotemporal Perspective</h1>

<div>
    <a href='#' target='_blank'>Zongheng Tang</a><sup>1,2</sup>&emsp;
    <a href='#' target='_blank'>Yi Liu</a><sup>2</sup>&emsp;
    <a href='#' target='_blank'>Yifan Sun</a><sup>2</sup>&emsp;
    <a href='#' target='_blank'>Yulu Gao</a><sup>1,2</sup>&emsp;
    <a href='#' target='_blank'>Jinyu Chen</a><sup>2</sup>&emsp;
    <a href='#' target='_blank'>Runsheng Xu</a><sup>3</sup>&emsp;
    <a href='#' target='_blank'>Si Liu</a><sup>2</sup>
</div>

<div>
    <sup>1</sup>Hangzhou International Innovation Institute, Beihang University&emsp;
    <sup>2</sup>School of Artificial Intelligence, Beihang University&emsp;
    <sup>3</sup>University of California, Los Angeles
</div>

<div>
    <strong>ICCV 2025 Highlight</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2508.00359" target='_blank'>
            <img src="https://img.shields.io/badge/arXiv-2508.00359-b31b1b.svg">
        </a>
        <a href="https://github.com/tzhhhh123/CoST" target='_blank'>
            <img src="https://img.shields.io/badge/Code-GitHub-black">
        </a>
        <a href="https://www.youtube.com/watch?v=DHre-cZSjAU" target='_blank'>
            <img src="https://img.shields.io/badge/Video-YouTube-red">
        </a>
        <a href="#citation" target='_blank'>
            <img src="https://img.shields.io/badge/Cite-BibTeX-blue">
        </a>
    </h4>
</div>

<strong>CoST is an efficient collaborative perception framework that unifies spatial and temporal collaboration among connected agents to improve perception accuracy while reducing communication cost.</strong>

<!--
<div style="text-align:center">
<img src="assets/teaser.png" width="100%" height="100%">
</div>
-->

> CoST aggregates observations from different agents and timestamps in a shared spatio-temporal space, enabling efficient feature transmission and unified spatio-temporal fusion.

---

</div>

## 📢 News

* **[2025-08-01]** 🔥 CoST paper is released on arXiv.
* **[2025-08-01]** 🏆 CoST is accepted as an ICCV 2025 Highlight paper.

## 💡 Highlights

* **Unified Spatiotemporal Perspective.** CoST jointly models spatial collaboration across agents and temporal collaboration across frames in a unified spatio-temporal space.
* **Efficiency-Oriented Design.** CoST reduces communication redundancy by avoiding repeated transmission of static object features while maintaining strong perception performance.
* **Flexible and General Framework.** CoST is compatible with many previous collaborative perception methods and can improve accuracy while reducing transmission bandwidth.
* **Multi-Dataset Validation.** CoST is evaluated on **V2V4Real**, **DAIR-V2X**, and **V2XSet**, demonstrating strong generalization across scenarios.

## 🛠️ Usage

This codebase is built upon [V2V4Real](https://github.com/ucla-mobility/V2V4Real). Please follow the instructions below to set up the environment, prepare datasets, and run training or evaluation.

### Installation

We recommend using `Conda` for environment management.

#### 1. Create Conda Environment

```bash
conda create -n v2v4real python=3.7 -y
conda activate v2v4real
```

#### 2. Install PyTorch

PyTorch `>= 1.12.0` is required. Example for CUDA 11.3:

```bash
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

#### 3. Install spconv 2.x

```bash
pip install spconv-cu113
```

#### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
python setup.py develop
```

#### 5. Build CUDA Extension for Bounding Box NMS

```bash
python opencood/utils/setup.py build_ext --inplace
```

#### 6. Install Deformable Convolution

```bash
cd opencood/models/sub_modules/ops
sh make.sh
```

### Data Preparation

CoST currently supports the following datasets:

* **V2V4Real**
* **DAIR-V2X**
* **V2XSet**

Please download the data in OPV2V format from the project website or the corresponding dataset pages. After downloading, organize the data as follows:

```shell
├── v2v4real
│   ├── train
│   │   ├── testoutput_CAV_data_2022-03-15-09-54-40_1
│   ├── validate
│   ├── test
```

### Model Preparation

Pre-trained checkpoints and model zoo links are not included in the current README. If you release checkpoints later, you can add them here, for example:

```text
checkpoints/
├── cost_v2v4real.pth
├── cost_dair_v2x.pth
└── cost_v2xset.pth
```

### Training

Run the provided training script:

```bash
bash train.sh
```

### Evaluation

Run the provided testing script:

```bash
bash test.sh
```

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{tang2025cost,
  title   = {CoST: Efficient Collaborative Perception From Unified Spatiotemporal Perspective},
  author  = {Tang, Zongheng and Liu, Yi and Sun, Yifan and Gao, Yulu and Chen, Jinyu and Xu, Runsheng and Liu, Si},
  journal = {arXiv preprint arXiv:2508.00359},
  year    = {2025}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License. See [LICENSE](./LICENSE) for more information.

## 🙏 Acknowledgement

This project is built upon [V2V4Real](https://github.com/ucla-mobility/V2V4Real). We thank the authors and contributors of the open-source collaborative perception community.


# CoST: Efficient Collaborative Perception From Unified Spatiotemporal Perspective

> This is the **official implementation** of the **ICCV 2025 Highlight paper**  
> **"CoST: Efficient Collaborative Perception From Unified Spatiotemporal Perspective"**  
> by [Zongheng Tang](#), [Yi Liu](#), [Yifan Sun](#), [Yulu Gao](#), [Jinyu Chen](#), [Runsheng Xu](#), and [Si Liu](#).

> 📄 Paper: [arXiv:2508.00359](https://arxiv.org/abs/2508.00359)  
> 🎥 Demo Video: [YouTube](https://www.youtube.com/watch?v=DHre-cZSjAU)

---

## ✨ Highlights

- **Unified Spatiotemporal Perspective:**  
  CoST introduces an efficient framework that jointly models spatial and temporal collaboration among connected agents.  
- **Efficiency-Oriented Design:**  
  Achieves high perception accuracy with reduced communication cost and computation overhead.  
- **Multi-dataset Validation:**  
  Verified on **V2V4Real**, **DAIR-V2X**, and **V2XSet** datasets, demonstrating strong generalization ability across scenarios.

---

## 📦 Supported Datasets

- **V2V4Real**
- **DAIR-V2X**
- **V2XSet**

### 📁 Data Download and Structure

Please check our website to download the data (OPV2V format).  
After downloading, organize the data as follows:

```shell
├── v2v4real
│   ├── train
│   │   ├── testoutput_CAV_data_2022-03-15-09-54-40_1
│   ├── validate
│   ├── test
```

---

## 🧰 Devkit Setup

CoST’s codebase is built upon [V2V4Real](https://github.com/ucla-mobility/V2V4Real).

Follow these steps to set up the environment:

#### 1. Create Conda Environment (Python ≥ 3.7)
```bash
conda create -n v2v4real python=3.7
conda activate v2v4real
```

#### 2. PyTorch Installation (≥ 1.12.0 Required)
Example for CUDA 11.3:
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

#### 3. spconv 2.x Installation
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

---

## 🚀 Quick Start

- **Training:**  
  Run the provided training script  
  ```bash
  bash train.sh
  ```

- **Testing:**  
  Run the provided testing script  
  ```bash
  bash test.sh
  ```

---

## 📜 License

This project is released under the **Apache 2.0 License**.  
See the [LICENSE](./LICENSE) file for details.

---

## 📚 Citation

If you find **CoST** useful in your research, please cite:

```bibtex
@article{tang2025cost,
  title   = {CoST: Efficient Collaborative Perception From Unified Spatiotemporal Perspective},
  author  = {Tang, Zongheng and Liu, Yi and Sun, Yifan and Gao, Yulu and Chen, Jinyu and Xu, Runsheng and Liu, Si},
  journal = {arXiv preprint arXiv:2508.00359},
  year    = {2025}
}
```

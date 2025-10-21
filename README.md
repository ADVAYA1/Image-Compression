# 🧠 Canonical Huffman Image Compressor

A **lossless grayscale image compression** project implemented using **Canonical Huffman Coding**.  
This program learns a general Huffman code table from training images, applies **inter-column differencing**, and achieves efficient, lossless compression with zero Mean Squared Error (MSE).

---

## 🚀 Features
- 📉 **Lossless compression** for grayscale `.tiff` and `.pgm` images  
- ⚡ **Canonical Huffman Code Table** generation from training data  
- 🧮 **Differencing-based preprocessing** for pixel decorrelation  
- 🔄 **Full pipeline:** training → compression → decompression  
- 🧾 **Performance metrics:** compression ratio & MSE validation  
- 💾 **Custom binary format** for compact storage

---

## 📂 Project Structure
canonical-huffman-image-compressor/
│
├── train/ # Training images (for generating Huffman codes)
├── val/ # Validation/test images
├── compressed/ # Output compressed files
├── reconstructed/ # Output reconstructed images
│
├── compressor.py # Main Python script (contains full implementation)
└── README.md # Project documentation

yaml
---

## ⚙️ Installation

### Prerequisites
Make sure you have **Python 3.8+** installed.  
Then install required dependencies:

```bash
pip install numpy pillow

▶️ Usage

Prepare your data:

Place sample grayscale images in:

train/ → used for Huffman table generation

val/ → used for compression & decompression testing

Run the compressor:

python compressor.py


Check results:

Compressed files → saved in compressed/

Reconstructed (decoded) images → saved in reconstructed/

Console output will show:

Original & compressed sizes

Compression ratio

MSE (Mean Squared Error)

📊 Example Output
--- Phase 1: Generating General Code Table ---
Analyzed 8 training images.
Generated code table with 295 entries.

--- Phase 2 & 3: Validating on 'val' folder ---

Processing: test_image.tiff
  Original Size:      524,288 bytes
  Compressed Size:    312,540 bytes
  Compression Ratio:  1.68:1
  MSE: 0.000000 ✓ LOSSLESS

🧩 Key Components
Function	Description
train()	Builds canonical Huffman codes from training images
compress()	Compresses input images using trained codes
decompress()	Reconstructs images from compressed files
_perform_differencing()	Applies pixel differencing for decorrelation
calculate_mse()	Calculates MSE to verify lossless reconstruction
🧪 Technical Highlights

Canonical Huffman implementation ensures deterministic, reproducible codes.

Out-of-bounds symbol handling for robust compression.

Uses NumPy for fast array math and PIL for image I/O.

Achieves perfect reconstruction (MSE = 0) on lossless images.

🪪 License

This project is licensed under the MIT License — you are free to use, modify, and distribute it.

⭐ Acknowledgements

Based on Huffman Coding concepts from data compression theory.

Implemented purely in Python using standard and open-source libraries.

❤️ If you find this useful, don’t forget to star the repo!

---

💡 **How to use:**
1. Copy all of the above text.  
2. Save it as `README.md` in your project root (same folder as your `.py` file).  
3. Commit and push:
   ```bash
   git add README.md
   git commit -m "Added project README"
   git push origin main
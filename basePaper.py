
import os
import struct
from collections import Counter
import numpy as np
from PIL import Image


PIXEL_RANGE_MIN = -150
PIXEL_RANGE_MAX = 151
OUT_OF_BOUNDS_SYMBOL = 'OOB' 

class CanonicalHuffmanCompressor:
    
    def __init__(self):
        
        self.code_table = {}
        self.decoding_tree = {}

    # ======================================================================
    # PHASE 1: OFFLINE CODE TABLE GENERATION
    # ======================================================================

    def train(self, train_dir):

        print("--- Phase 1: Generating General Code Table ---")
        
        master_freq_counter = Counter()
        image_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".tiff", ".tif", ".pgm"))]
        
        if not image_files:
            raise FileNotFoundError(f"No training images found in directory: {train_dir}")

        for filename in image_files:
            filepath = os.path.join(train_dir, filename)
            image_array = self._read_image(filepath)
            if image_array is not None:
                _, diff_data = self._perform_differencing(image_array)
                if diff_data is not None:
                    in_range_pixels = [p for p in diff_data if PIXEL_RANGE_MIN <= p <= PIXEL_RANGE_MAX]
                    master_freq_counter.update(in_range_pixels)
        
        print(f"Analyzed {len(image_files)} training images.")
        
        master_freq_counter[OUT_OF_BOUNDS_SYMBOL] = 1
        
        N_model = [[symbol, freq, 0] for symbol, freq in master_freq_counter.items()]
        
        # Algorithm 1: Build the Huffman tree structure H
        H_tree = self._canonical_huffman_tree_creation(N_model)
        
        # Algorithm 2: Calculate code lengths and update N_model
        N_model = self._calculate_code_lengths(N_model, H_tree)
        
        # Algorithm 3: Count codes of each length to create the C array
        C_array = self._calculate_C_array(N_model)
        
        # Sort N_model for deterministic code assignment (by length, then symbol)
        N_sorted = sorted(N_model, key=lambda x: (x[2], x[0] if isinstance(x[0], int) else float('inf')))
        
        # Generate the final code table from the C array and sorted symbols
        self.code_table = self._generate_codes_from_C(C_array, N_sorted)
        
        # Build a decoding tree (Trie) for fast decompression
        self.decoding_tree = self._build_decoding_tree()
        
        print(f"Generated code table with {len(self.code_table)} entries.")

    def _canonical_huffman_tree_creation(self, N_model):

        n = len(N_model)
        H = np.zeros((2 * n - 1, 2), dtype=np.int64)
        
        for i in range(n):
            H[i, 0] = N_model[i][1]
        
        next_node_idx = n
        for _ in range(n - 1):
            min1_val, min2_val = float('inf'), float('inf')
            min1_idx, min2_idx = -1, -1
            
            for i in range(next_node_idx):
                if H[i, 1] == 0:
                    if H[i, 0] < min1_val:
                        min2_val, min2_idx = min1_val, min1_idx
                        min1_val, min1_idx = H[i, 0], i
                    elif H[i, 0] < min2_val:
                        min2_val, min2_idx = H[i, 0], i
            
            H[next_node_idx, 0] = H[min1_idx, 0] + H[min2_idx, 0]
            H[min1_idx, 1] = next_node_idx
            H[min2_idx, 1] = next_node_idx
            next_node_idx += 1
        return H

    def _calculate_code_lengths(self, N_model, H_tree):

        n = len(N_model)
        for i in range(n):
            code_length = 0
            parent_idx = int(H_tree[i, 1])
            while parent_idx != 0:
                code_length += 1
                parent_idx = int(H_tree[parent_idx, 1])
            N_model[i][2] = code_length
        return N_model

    def _calculate_C_array(self, N_model):
      
        max_len = max(item[2] for item in N_model) if N_model else 0
        C = np.zeros(max_len + 1, dtype=np.int32)
        for _, _, length in N_model:
            if length > 0:
                C[length] += 1
        return C

    def _generate_codes_from_C(self, C_array, N_sorted):
     
        codes = {}
        code_val = 0
        n_idx = 0
        
        for length in range(1, len(C_array)):
            for _ in range(C_array[length]):
                symbol = N_sorted[n_idx][0]
                code_str = format(code_val, f'0{length}b')
                codes[symbol] = code_str
                code_val += 1
                n_idx += 1
            code_val <<= 1
        return codes

    def _build_decoding_tree(self):
        tree = {}
        for symbol, code_str in self.code_table.items():
            node = tree
            for bit in code_str:
                if bit not in node:
                    node[bit] = {}
                node = node[bit]
            node['value'] = symbol
        return tree

    # ======================================================================
    # PHASE 2: REAL-TIME COMPRESSION (Algorithm 4)
    # ======================================================================

    def compress(self, in_path, out_path):
    
        if not self.code_table:
            raise RuntimeError("Compressor has not been trained. Call train() first.")

        original_array = self._read_image(in_path)
        if original_array is None: return

        height, width = original_array.shape
        first_column, diff_data = self._perform_differencing(original_array)
        
        out_of_bounds_pixels = []
        bits_list = []
        oob_code = self.code_table[OUT_OF_BOUNDS_SYMBOL]
        
        for pixel_diff in diff_data:
            if PIXEL_RANGE_MIN <= pixel_diff <= PIXEL_RANGE_MAX:
                bits_list.append(self.code_table[pixel_diff])
            else:
                bits_list.append(oob_code)
                out_of_bounds_pixels.append(pixel_diff)
        
        bit_string = "".join(bits_list)
        
        padding_needed = (8 - len(bit_string) % 8) % 8
        bit_string_padded = bit_string + '0' * padding_needed
        byte_array = bytearray(int(bit_string_padded[i:i+8], 2) for i in range(0, len(bit_string_padded), 8))
        
        with open(out_path, 'wb') as f:
            f.write(struct.pack('<IIQ', height, width, len(bit_string)))
            first_column_bytes = first_column.astype(np.int32).tobytes()
            f.write(struct.pack('<I', len(first_column_bytes)))
            f.write(first_column_bytes)
            f.write(struct.pack('<I', len(out_of_bounds_pixels)))
            if out_of_bounds_pixels:
                f.write(np.array(out_of_bounds_pixels, dtype=np.int32).tobytes())
            f.write(byte_array)

    # ======================================================================
    # PHASE 3: REAL-TIME DECOMPRESSION
    # ======================================================================

    def decompress(self, in_path):
      
        if not self.decoding_tree:
            raise RuntimeError("Compressor has not been trained. Call train() first.")

        with open(in_path, 'rb') as f:
            height, width, original_bit_count = struct.unpack('<IIQ', f.read(16))
            first_col_len = struct.unpack('<I', f.read(4))[0]
            first_column = np.frombuffer(f.read(first_col_len), dtype=np.int32)
            oob_count = struct.unpack('<I', f.read(4))[0]
            oob_pixels = np.frombuffer(f.read(oob_count * 4), dtype=np.int32).tolist() if oob_count > 0 else []
            byte_array = f.read()
        
        bit_string = "".join(f"{byte:08b}" for byte in byte_array)[:original_bit_count]
        
        diff_data = []
        oob_idx = 0
        node = self.decoding_tree
        for bit in bit_string:
            node = node[bit]
            if 'value' in node:
                value = node['value']
                if value == OUT_OF_BOUNDS_SYMBOL:
                    diff_data.append(oob_pixels[oob_idx])
                    oob_idx += 1
                else:
                    diff_data.append(value)
                node = self.decoding_tree
        
        reconstructed_image = np.zeros((height, width), dtype=np.int32)
        reconstructed_image[:, 0] = first_column
        diff_data_reshaped = np.array(diff_data, dtype=np.int32).reshape(height, width - 1)
        
        for j in range(1, width):
            reconstructed_image[:, j] = reconstructed_image[:, j - 1] + diff_data_reshaped[:, j - 1]
        
        np.clip(reconstructed_image, 0, 65535, out=reconstructed_image)
        return reconstructed_image.astype(np.uint16)


    @staticmethod
    def _read_image(path):
        """Reads a TIFF or PGM image and returns it as a NumPy array."""
        try:
            img = Image.open(path)
            if img.mode == 'I;16B':
                return np.array(img, dtype=np.uint16).byteswap()
            return np.array(img, dtype=np.uint16)
        except Exception as e:
            print(f"Warning: Could not read image {os.path.basename(path)}. Error: {e}")
            return None

    @staticmethod
    def _perform_differencing(image_array):
        """
        Implements inter-column differencing as per Formula (3): D(i,j) = p(i,j) - p(i,j-1).
        """
        height, width = image_array.shape
        image_array_int = image_array.astype(np.int32)
        first_column = image_array_int[:, 0].copy()
        diff_image = image_array_int[:, 1:] - image_array_int[:, :-1]
        return first_column, diff_image.flatten()

# ======================================================================
# UTILITY FUNCTIONS AND MAIN EXECUTION
# ======================================================================

def calculate_mse(original, reconstructed):
    """Calculates the Mean Squared Error (MSE) as per Formula."""
    if original.shape != reconstructed.shape:
        print(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
        return -1.0
    return np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)

def save_image(image_array, output_path):
    """Saves a NumPy array as a TIFF image."""
    Image.fromarray(image_array).save(output_path)

def main():
    """Main execution block to run the compression pipeline."""
    train_dir = "train"
    val_dir = "val"
    output_dir = "compressed"
    recon_dir = "reconstructed"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    compressor = CanonicalHuffmanCompressor()
    try:
        compressor.train(train_dir)
    except Exception as e:
        print(f"Fatal Error during training: {e}")
        return

    print(f"\n--- Phase 2 & 3: Validating on '{val_dir}' folder ---")
    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith((".tiff", ".tif", ".pgm"))]
    if not val_files:
        print("No validation images found. Exiting.")
        return

    for filename in sorted(val_files):
        in_path = os.path.join(val_dir, filename)
        out_filename = os.path.splitext(filename)[0] + ".compressed"
        out_path = os.path.join(output_dir, out_filename)
        recon_path = os.path.join(recon_dir, filename)
        
        print(f"\nProcessing: {filename}")
        
        original_array = CanonicalHuffmanCompressor._read_image(in_path)
        if original_array is None: continue
        
        compressor.compress(in_path, out_path)
        reconstructed_array = compressor.decompress(out_path)
        save_image(reconstructed_array, recon_path)
        
        original_size = os.path.getsize(in_path)
        compressed_size = os.path.getsize(out_path)
        mse = calculate_mse(original_array, reconstructed_array)
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        print(f"  Original Size:      {original_size:,} bytes")
        print(f"  Compressed Size:    {compressed_size:,} bytes")
        print(f"  Compression Ratio:  {ratio:.2f}:1")
        print(f"  MSE: {mse:.6f} {'✓ LOSSLESS' if mse == 0.0 else '✗ LOSSY'}")

if __name__ == "__main__":
    main()
import streamlit as st
import heapq
from collections import defaultdict

def huffman_codes(freq):
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def char_to_binary(char):
    binary_rep = bin(ord(char))[2:]
    return '0'*(8-len(binary_rep)) + binary_rep

def main():
    st.title("Konversi Karakter ke Biner dan Huffman Codes")

    text_input = st.text_input("Masukkan teks:")
    
    if not text_input:
        st.warning("Masukkan teks terlebih dahulu.")
        return

    freq = defaultdict(int)
    for char in text_input:
        freq[char] += 1

    binary_output = [char_to_binary(char) for char in text_input]
    huffman_output = huffman_codes(freq)

    st.header("Frekuensi Karakter:")
    freq_table_data = {'Karakter': list(freq.keys()), 'Weight': list(freq.values()), 'Frekuensi': [f'{count / len(text_input):.2%}' for count in freq.values()]}
    st.table(freq_table_data)

    st.header("Konversi Karakter ke Biner dan Huffman Codes:")
    combined_data = {'Karakter': list(text_input), 'Biner': binary_output, 'Huffman Code': [''] * len(text_input)}

    for char, huffman_code in huffman_output:
        indices = [i for i, c in enumerate(text_input) if c == char]
        for index in indices:
            combined_data['Huffman Code'][index] = huffman_code

    st.table(combined_data)

if __name__ == "__main__":
    main()

'''
Запуск в терминале python decompress.py
Необходимо разорхивировать файл из nyuuzyou/wb-products
'''
import zstandard as zstd
input_path_products = "basket-01.json.zst"
output_path_products = "basket-01.json"
with open(input_path_products, "rb") as compressed_file_products_1:
    dctx = zstd.ZstdDecompressor()
    with open(output_path_products, "wb") as output_file_products_1:
        dctx.copy_stream(compressed_file_products_1, output_file_products_1)

import zstandard as zstd
input_path_products = "basket-07.json.zst"
output_path_products = "basket-07.json"
with open(input_path_products, "rb") as compressed_file_products_2:
    dctx = zstd.ZstdDecompressor()
    with open(output_path_products, "wb") as output_file_products_2:
        dctx.copy_stream(compressed_file_products_2, output_file_products_2)

'''
Необходимо разорхивировать файл из nyuuzyou/wb-feedbacks
'''
input_path_feedbacks = "feedbacks-08.json.zst"
output_path_feedbacks = "feedbacks-08.json"
with open(input_path_feedbacks, "rb") as compressed_file_feedbacks:
    dctx = zstd.ZstdDecompressor()
    with open(output_path_feedbacks, "wb") as output_file_feedbacks:
        dctx.copy_stream(compressed_file_feedbacks, output_file_feedbacks)
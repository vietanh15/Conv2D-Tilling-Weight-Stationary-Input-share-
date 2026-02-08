IFM: params/ifm.txt ,shape   : [1, 112, 112, 32]
WEIGHTS: params/weights.txt ,shape   : [1, 3, 3, 32]
OFM golden: golden_output/ofm_golden.txt ,shape   : [1, 112, 112, 1]
Stride=(1,1)

Viết lại vòng for trong phép Conv2D: có lệnh load, store, tính toán, share buffer.
Sắp xếp để tính toán sử dụng các phương pháp: tiling, weight stationary, input share.

Code bằng C -> đo latency, sử dụng bao nhiêu memory
Để chạy file so sánh cần thay đổi đường dẫn của ofm (vì có 3 phương pháp khác nhau) trong 2_sosanhxuatpho.py"# Conv2D-Tilling-Weight-Stationary-Input-share-" 

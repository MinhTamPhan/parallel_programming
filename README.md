# parallel_programming

# Hướng đẫn
* Tuần này cố gắng cài lại và hiểu các bài toán scan và tính histogram
    * Nên xem lại video của thầy để làm thêm các cải tiến.
    * Cố gắng cài lại thuật toán sort của thầy chạy trên host. Xem lại bài giảng.
    * Có thời gian đọc thêm các bài báo ở thư mục papers

* Một số link tham khảo:
    * `https://github.com/mark-poscablo/gpu-radix-sort/tree/master/radix_sort`
    * `https://github.com/moderngpu/moderngpu/tree/master/src/moderngpu`
    * `https://github.com/NVlabs/cub`
    * `https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting`
    * `https://github.com/deeperlearning/professional-cuda-c-programming`

* Cấu trúc thư mục:
    * Khi code 1 bài mới ví dụ, scan. thì tạo 1 file `scan.cu`, `hits.cu` ... sau đó include ```#include "helper.cuh"``` vào file mới trong này chứ hàm `CHECK` để check `GPU` và struct `GpuTimer`. Sau đó code các hàm cần thiết không code hàm main vào các file này.
    * Sau đó vào thư mục test. tạo file test tương tự ví dụ `scan_test.cu` sau đó code lại hàm main nhớ include các file cần thiết ví dụ `#include "../src/scan.cu"`

* Để chạy code này cần upload các file trong thư mục `src` vào thực mục `src`, `testing` tạo trên colab và upload toàn bộ các file trong thư mục đó. ví dụ như hình sau:

![hinh 1](./colab.png)

- sau đó chạy lệnh sau trên colab

`!nvcc testing/scan_test.cu && ./a.out`

có thể tham khảo file [example](./example.ipynb)

## Tuần 13/9 - 19/9:
* 14/9 : làm thử phiên bản baseline 2: 
    - tính song song hóa quá trình tính hist (done)-phiên bản bình thường. đơn giản chỉnh sửa phần tính hist dúng các bit cần tính. thử nghiệm 2 kernal sử dụng SMEM và không sử dụng SMEM.
        - Nhận thấy việc sử dụng SMEM hoặc không sử dụng SMEM không ảnh hưởng quá nhiều với quá trình sorting với blocksize = `256, 512` (k nhận thấy sự chênh lệnh thời gian quá nhiều, có thể là do nBins = 4 khá nhỏ nên việc sử dụng SMEM k đem lại hiệu quả)
        - TODO chạy thử trên colab
    - song song quá trính scan (inprocess) - TODO


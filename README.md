# fft-convolution

Convolve an image with a kernel matrix (csv-file).

Specify relative path to input image, kernel and output image

Example:

```
cargo run --release input/ole_ivars.jpg gaussian_kernels/sigma_64.csv output/ole_ivars_blurred.jpg
```
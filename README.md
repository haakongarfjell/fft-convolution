# fft-convolution

Convolve an image with a gaussian kernel using the fast fourier transform. 

Specify relative path to input image, output image and gaussian variance $\sigma$.

Example:

```
cargo run --release input/ole_ivars.jpg output/ole_ivars_blurred.jpg 10
```
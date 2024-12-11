use std::f64::consts::PI;
use std::env;

use image::{ImageBuffer, ImageReader, Rgb};

#[derive(Clone, Copy)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    
    fn add(&self, other: &Complex) -> Complex {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }

    
    fn sub(&self, other: &Complex) -> Complex {
        Complex::new(self.real - other.real, self.imag - other.imag)
    }

    
    fn mul(&self, other: &Complex) -> Complex {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }

    fn div(&self, scalar: f64) -> Complex {
        Complex::new(self.real / scalar, self.imag / scalar)
    }
}


fn pad_matrix(matrix: &mut Vec<Vec<Complex>>) -> (usize, usize) {
    let original_rows = matrix.len();
    let original_cols = matrix[0].len();
    let target_rows = original_rows.next_power_of_two();
    let target_cols = original_cols.next_power_of_two();

    let row_padding = (target_rows - original_rows) / 2;
    let col_padding = (target_cols - original_cols) / 2;

   
    let mut padded_matrix = vec![vec![Complex::new(0.0, 0.0); target_cols]; target_rows];

    for i in 0..original_rows {
        for j in 0..original_cols {
            padded_matrix[row_padding + i][col_padding + j] = matrix[i][j];
        }
    }

    *matrix = padded_matrix;

    (original_rows, original_cols)
}

fn unpad_matrix(matrix: &mut Vec<Vec<Complex>>, original_rows: usize, original_cols: usize) {
    let target_rows = matrix.len();
    let target_cols = matrix[0].len();

    let row_padding = (target_rows - original_rows) / 2;
    let col_padding = (target_cols - original_cols) / 2;

    let mut unpadded_matrix = vec![vec![Complex::new(0.0, 0.0); original_cols]; original_rows];

    for i in 0..original_rows {
        for j in 0..original_cols {
            unpadded_matrix[i][j] = matrix[row_padding + i][col_padding + j];
        }
    }

    *matrix = unpadded_matrix;
}


/// Perform a 1D FFT (Radix-2 DIT) on a vector of Complex numbers
fn fft_1d(data: &mut Vec<Complex>, inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    
    let mut even = vec![Complex::new(0.0, 0.0); n / 2];
    let mut odd = vec![Complex::new(0.0, 0.0); n / 2];
    for i in 0..n / 2 {
        even[i] = data[i * 2];
        odd[i] = data[i * 2 + 1];
    }

    
    fft_1d(&mut even, inverse);
    fft_1d(&mut odd, inverse);

    let angle = if inverse {
        2.0 * PI / n as f64 * 1.0
    } else {
        2.0 * PI / n as f64 * -1.0
    };

    let w_n = Complex::new(angle.cos(), angle.sin());
    let mut w = Complex::new(1.0, 0.0);
    for i in 0..n / 2 {
        let t = w.mul(&odd[i]);
        data[i] = even[i].add(&t);
        data[i + n / 2] = even[i].sub(&t);
        w = w.mul(&w_n);
    }
}

fn fft_2d(matrix: &mut Vec<Vec<Complex>>, inverse: bool) {
    let rows = matrix.len();
    let cols = matrix[0].len();

    // Row FFT
    for row in matrix.iter_mut() {
        fft_1d(row, inverse);
    }

    
    let mut transposed = vec![vec![Complex::new(0.0, 0.0); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    // Column FFT
    for row in transposed.iter_mut() {
        fft_1d(row, inverse);
    }

    
    for i in 0..cols {
        for j in 0..rows {
            matrix[j][i] = transposed[i][j];
        }
    }

    
    if inverse {
        let norm_factor = (rows * cols) as f64;
        for row in matrix.iter_mut() {
            for val in row.iter_mut() {
                *val = val.div(norm_factor);
            }
        }
    }
}

fn load_image(path: &str) -> (Vec<Vec<Complex>>, Vec<Vec<Complex>>, Vec<Vec<Complex>>) {
    let img = ImageReader::open(path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (cols, rows) = img.dimensions();
    let mut red_channel = vec![vec![Complex::new(0.0, 0.0); cols as usize]; rows as usize];
    let mut green_channel = vec![vec![Complex::new(0.0, 0.0); cols as usize]; rows as usize];
    let mut blue_channel = vec![vec![Complex::new(0.0, 0.0); cols as usize]; rows as usize];

    for y in 0..rows as usize {
        for x in 0..cols as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            red_channel[y][x] = Complex::new(pixel[0] as f64 / 255.0, 0.0); // Normalize to [0, 1]
            green_channel[y][x] = Complex::new(pixel[1] as f64 / 255.0, 0.0); // Normalize to [0, 1]
            blue_channel[y][x] = Complex::new(pixel[2] as f64 / 255.0, 0.0); // Normalize to [0, 1]
        }
    }

    (red_channel, green_channel, blue_channel)
}

fn save_image(path: &str, red_channel: &Vec<Vec<Complex>>, green_channel: &Vec<Vec<Complex>>, blue_channel: &Vec<Vec<Complex>>) {
    let rows = red_channel.len();
    let cols = red_channel[0].len();

    let mut img_buffer = ImageBuffer::new(cols as u32, rows as u32);

    for y in 0..rows {
        for x in 0..cols {
            
            let r = (red_channel[y][x].real * 255.0).clamp(0.0, 255.0) as u8;
            let g = (green_channel[y][x].real * 255.0).clamp(0.0, 255.0) as u8;
            let b = (blue_channel[y][x].real * 255.0).clamp(0.0, 255.0) as u8;

            
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    
    let _ = img_buffer.save(path);
}

fn generate_gaussian_kernel(sigma: f32) -> Vec<Vec<f32>> {
    let size = 2 * (sigma * 3.0) as usize + 1;
    let mut kernel = vec![vec![0.0; size]; size];
    let center = size as isize / 2;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for i in 0..size {
        for j in 0..size {
            let x = i as isize - center;
            let y = j as isize - center;
            kernel[i][j] = (-((x * x + y * y) as f32) / two_sigma_sq).exp();
            sum += kernel[i][j];
        }
    }

    // Normalize
    for i in 0..size {
        for j in 0..size {
            kernel[i][j] /= sum;
        }
    }

    println!("Generated Gaussian Kernel with sigma = {} and size = {}", sigma, size);

    kernel
}

fn fftshift(matrix: &mut Vec<Vec<Complex>>) {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let half_rows = rows / 2;
    let half_cols = cols / 2;

    for i in 0..half_rows {
        for j in 0..half_cols {
            // Swap Top-left Bottom-right
            let tmp = matrix[i][j];
            matrix[i][j] = matrix[i + half_rows][j + half_cols];
            matrix[i + half_rows][j + half_cols] = tmp;

            // Swap Top-right Bottom-left
            let tmp = matrix[i][j + half_cols];
            matrix[i][j + half_cols] = matrix[i + half_rows][j];
            matrix[i + half_rows][j] = tmp;
        }
    }
}

fn center_and_pad_kernel(kernel: Vec<Vec<f32>>, target_rows: usize, target_cols: usize) -> Vec<Vec<Complex>> {
    let kernel_size = kernel.len();

    let mut padded_kernel = vec![vec![Complex::new(0.0, 0.0); target_cols]; target_rows];

    // Center the kernel in the padded space
    let start_row = (target_rows - kernel_size) / 2;
    let start_col = (target_cols - kernel_size) / 2;

    for i in 0..kernel_size {
        for j in 0..kernel_size {
            padded_kernel[start_row + i][start_col + j] = Complex::new(kernel[i][j] as f64, 0.0);
        }
    }
    

    padded_kernel
}

fn convolve(channel: &mut Vec<Vec<Complex>>, kernel: &Vec<Vec<Complex>>) {
    let rows = channel.len();
    let cols = channel[0].len();

    for i in 0..rows {
        for j in 0..cols {
            channel[i][j] = channel[i][j].mul(&kernel[i][j]);
        }
    }
}

fn main() {
    let input_path = env::args().nth(1).expect("Provide relative path to image");
    let output_path = env::args().nth(2).expect("Provide relative path to output image");
    let sigma: f32 = env::args()
        .nth(3)
        .expect("Provide sigma value.")
        .parse()
        .expect("Enter a valid value.");

    let current_dir = env::current_dir().expect("Failed to get directory");

    let input_path_full = current_dir.join(input_path);
    let output_path_full = current_dir.join(output_path);

    let (mut red_channel, mut green_channel, mut blue_channel) = load_image(input_path_full.to_str().unwrap());




    let (original_rows_r, original_cols_r) = pad_matrix(&mut red_channel);
    let (original_rows_g, original_cols_g) = pad_matrix(&mut green_channel);
    let (original_rows_b, original_cols_b) = pad_matrix(&mut blue_channel);

    let gaussian_kernel = generate_gaussian_kernel(sigma);

    let mut center_padded_kernel = center_and_pad_kernel(gaussian_kernel, original_rows_r, original_cols_r);

    let (_orignal_rows_kernel, _original_cols_kernel) = pad_matrix(&mut center_padded_kernel);


    fftshift(&mut center_padded_kernel);
    fft_2d(&mut center_padded_kernel, false);


    fft_2d(&mut red_channel, false);
    fft_2d(&mut green_channel, false);
    fft_2d(&mut blue_channel, false);

    convolve(&mut red_channel, &center_padded_kernel);
    convolve(&mut green_channel, &center_padded_kernel);
    convolve(&mut blue_channel, &center_padded_kernel);

    fft_2d(&mut red_channel, true);
    fft_2d(&mut green_channel, true);
    fft_2d(&mut blue_channel, true);

    unpad_matrix(&mut red_channel, original_rows_r, original_cols_r);
    unpad_matrix(&mut green_channel, original_rows_g, original_cols_g);
    unpad_matrix(&mut blue_channel, original_rows_b, original_cols_b);

    let _ = save_image(output_path_full.to_str().unwrap(), &red_channel, &green_channel, &blue_channel);

}
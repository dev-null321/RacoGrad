#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Convolution operation (naive implementation)
void conv2d_forward(int batch_size, int in_channels, int in_height, int in_width,
                    int out_channels, int filter_height, int filter_width,
                    int stride, int padding,
                    double* input, double* filters, double* output) {
    
    int out_height = 1 + (in_height + 2 * padding - filter_height) / stride;
    int out_width = 1 + (in_width + 2 * padding - filter_width) / stride;
    
    // Initialize output to zeros
    int output_size = batch_size * out_channels * out_height * out_width;
    memset(output, 0, output_size * sizeof(double));

    // For each element in the batch
    for (int b = 0; b < batch_size; b++) {
        // For each output channel
        for (int oc = 0; oc < out_channels; oc++) {
            // For each input channel
            for (int ic = 0; ic < in_channels; ic++) {
                // For each output position
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        // Calculate output index
                        int out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        
                        // For each filter position
                        for (int fh = 0; fh < filter_height; fh++) {
                            for (int fw = 0; fw < filter_width; fw++) {
                                // Calculate input position with padding
                                int ih = oh * stride + fh - padding;
                                int iw = ow * stride + fw - padding;
                                
                                // Skip if outside the input boundaries
                                if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
                                    continue;
                                }
                                
                                // Calculate input index
                                int in_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                
                                // Calculate filter index
                                int filter_idx = ((oc * in_channels + ic) * filter_height + fh) * filter_width + fw;
                                
                                // Add contribution to output
                                output[out_idx] += input[in_idx] * filters[filter_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Max pooling 2x2 with stride 2
void max_pool_2x2(int batch_size, int channels, int in_height, int in_width,
                 double* input, double* output) {
    
    int out_height = in_height / 2;
    int out_width = in_width / 2;
    
    // Initialize output to very negative values
    int output_size = batch_size * channels * out_height * out_width;
    for (int i = 0; i < output_size; i++) {
        output[i] = -1e9;
    }
    
    // For each element in the batch
    for (int b = 0; b < batch_size; b++) {
        // For each channel
        for (int c = 0; c < channels; c++) {
            // For each pooling region
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    // Calculate output index
                    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    
                    // For each input in the pooling region
                    for (int kh = 0; kh < 2; kh++) {
                        for (int kw = 0; kw < 2; kw++) {
                            // Calculate input index
                            int ih = oh * 2 + kh;
                            int iw = ow * 2 + kw;
                            int in_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                            
                            // Update max value
                            if (input[in_idx] > output[out_idx]) {
                                output[out_idx] = input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Flatten a 4D tensor (batch, channels, height, width) to a 2D tensor (batch, channels*height*width)
void flatten_tensor(int batch_size, int channels, int height, int width,
                   double* input, double* output) {
    
    int flat_size = channels * height * width;
    
    // For each batch element
    for (int b = 0; b < batch_size; b++) {
        // For each channel
        for (int c = 0; c < channels; c++) {
            // For each height
            for (int h = 0; h < height; h++) {
                // For each width
                for (int w = 0; w < width; w++) {
                    // Calculate input index
                    int in_idx = ((b * channels + c) * height + h) * width + w;
                    
                    // Calculate flat index
                    int flat_idx = b * flat_size + (c * height * width + h * width + w);
                    
                    // Copy value
                    output[flat_idx] = input[in_idx];
                }
            }
        }
    }
}

// Softmax operation
void softmax(int batch_size, int num_classes, double* input, double* output) {
    // For each batch
    for (int b = 0; b < batch_size; b++) {
        // Find max value for stability
        double max_val = -1e9;
        for (int c = 0; c < num_classes; c++) {
            if (input[b * num_classes + c] > max_val) {
                max_val = input[b * num_classes + c];
            }
        }
        
        // Calculate exp and sum
        double sum = 0.0;
        for (int c = 0; c < num_classes; c++) {
            output[b * num_classes + c] = exp(input[b * num_classes + c] - max_val);
            sum += output[b * num_classes + c];
        }
        
        // Normalize
        for (int c = 0; c < num_classes; c++) {
            output[b * num_classes + c] /= sum;
        }
    }
}

// Cross entropy loss
double cross_entropy_loss(int batch_size, int num_classes, double* pred, double* target) {
    double loss = 0.0;
    double epsilon = 1e-15;  // Small value for numerical stability
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_classes; c++) {
            // Clamp prediction for numerical stability
            double p = pred[b * num_classes + c];
            if (p < epsilon) p = epsilon;
            if (p > 1.0 - epsilon) p = 1.0 - epsilon;
            
            // Add to loss if target is 1
            if (target[b * num_classes + c] > 0.5) {
                loss -= log(p);
            }
        }
    }
    
    return loss / batch_size;
}
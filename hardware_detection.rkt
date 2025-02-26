#lang racket

(require ffi/unsafe)

(provide detect-hardware-capabilities
         get-optimal-num-threads
         has-avx?
         has-sse?
         has-opencl?
         is-apple-silicon?
         has-mlx-support?
         has-cuda-support?
         print-hardware-info)

;; Safely try to load the libraries with proper error handling
(define libcheck 
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "simd_ops" '("" "0"))))

;; Safely try to load the libraries and their availability checks
(define opencl-lib 
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "matrix_opencl" '("" "0"))))

(define mlx-lib 
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "mlx_ops" '("" "0"))))

(define cuda-lib 
  (with-handlers ([exn:fail? (lambda (e) #f)])
    (ffi-lib "cuda_ops" '("" "0"))))

;; Import availability check functions if libraries are available
(define check-opencl-available
  (if opencl-lib
      (with-handlers ([exn:fail? (lambda (e) (lambda () 0))])
        (get-ffi-obj "check_opencl_available" opencl-lib (_fun -> _int)))
      (lambda () 0)))

(define check-mlx-available
  (if mlx-lib
      (with-handlers ([exn:fail? (lambda (e) (lambda () 0))])
        (get-ffi-obj "check_mlx_available" mlx-lib (_fun -> _int)))
      (lambda () 0)))

(define check-cuda-available
  (if cuda-lib
      (with-handlers ([exn:fail? (lambda (e) (lambda () 0))])
        (get-ffi-obj "check_cuda_available" cuda-lib (_fun -> _int)))
      (lambda () 0)))

;; Number of physical cores detection
(define (detect-num-cores)
  (cond
    [(equal? (system-type 'os) 'unix)
     (with-handlers ([exn:fail? (lambda (_) 4)]) ; Default to 4 if detection fails
       (let* ([output (with-output-to-string
                        (lambda () (system "sysctl -n hw.physicalcpu 2>/dev/null || 
                                            grep -c ^processor /proc/cpuinfo 2>/dev/null || 
                                            echo 4")))]
              [num (string->number (string-trim output))])
         (if num num 4)))]
    [(equal? (system-type 'os) 'windows)
     (with-handlers ([exn:fail? (lambda (_) 4)])
       (let* ([output (with-output-to-string
                        (lambda () (system "echo %NUMBER_OF_PROCESSORS%")))]
              [num (string->number (string-trim output))])
         (if num num 4)))]
    [else 4])) ; Default value

;; Check for SIMD instruction support
(define (has-simd-support? type)
  (cond
    [(equal? (system-type 'os) 'unix)
     (case type
       [(avx) (with-handlers ([exn:fail? (lambda (_) #f)])
               (let ([output (with-output-to-string
                              (lambda () (system "grep -q avx /proc/cpuinfo && echo yes || 
                                                  sysctl -n machdep.cpu.features 2>/dev/null | grep -q AVX && echo yes")))])
                 (string-contains? (string-trim output) "yes")))]
       [(sse) (with-handlers ([exn:fail? (lambda (_) #f)])
               (let ([output (with-output-to-string
                              (lambda () (system "grep -q sse /proc/cpuinfo && echo yes || 
                                                  sysctl -n machdep.cpu.features 2>/dev/null | grep -q SSE && echo yes")))])
                 (string-contains? (string-trim output) "yes")))]
       [else #f])]
    [else #f])) ; Simplified for other OSes

;; Check for OpenCL support
(define (has-opencl-support?)
  (and opencl-lib (= (check-opencl-available) 1)))

;; Check for Apple Silicon (M1/M2/M3)
(define (detect-apple-silicon)
  (and (equal? (system-type 'os) 'macosx)
       (or (string-contains? (with-output-to-string 
                              (lambda () (system "uname -m")))
                             "arm64")
           (string-contains? (with-output-to-string 
                              (lambda () (system "sysctl -n machdep.cpu.brand_string 2>/dev/null")))
                             "Apple"))))

;; Check for MLX availability on Apple Silicon
(define (has-mlx-support?)
  (and (detect-apple-silicon)
       ;; Check for MLX library in standard locations
       (or mlx-lib 
           (with-handlers ([exn:fail? (lambda (e) #f)])
             (file-exists? "/opt/homebrew/lib/libmlx.dylib"))
           (with-handlers ([exn:fail? (lambda (e) #f)])
             (file-exists? "/usr/local/lib/libmlx.dylib"))
           (with-handlers ([exn:fail? (lambda (e) #f)])
             (file-exists? "/usr/lib/libmlx.dylib")))
       ;; Fall back to environment variable check
       (or (= (check-mlx-available) 1)
           (getenv "MLX_AVAILABLE"))))

;; Check for CUDA availability
(define (has-cuda-support?)
  (and cuda-lib 
       (= (check-cuda-available) 1)))

;; Memory size detection (in GB)
(define (detect-memory-size)
  (cond
    [(equal? (system-type 'os) 'unix)
     (with-handlers ([exn:fail? (lambda (_) 8.0)]) ; Default to 8GB
       (let* ([output (with-output-to-string
                        (lambda () (system "sysctl -n hw.memsize 2>/dev/null || 
                                            grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || 
                                            echo 8589934592")))]
              [num (string->number (string-trim output))])
         (if num
             (if (< num 1000000) ; If returned in KB (Linux)
                 (/ num 1024 1024)
                 (/ num 1024 1024 1024))
             8.0)))]
    [else 8.0])) ; Default for other OSes

;; Main hardware detection function
(define (detect-hardware-capabilities)
  (let ([cores (detect-num-cores)]
        [avx (has-simd-support? 'avx)]
        [sse (has-simd-support? 'sse)]
        [opencl (has-opencl-support?)]
        [apple-silicon (detect-apple-silicon)]
        [mlx (has-mlx-support?)]
        [cuda (has-cuda-support?)]
        [memory (detect-memory-size)])
    (hash 'cores cores
          'avx avx
          'sse sse
          'opencl opencl
          'apple-silicon apple-silicon
          'mlx mlx
          'cuda cuda
          'memory memory)))

;; Cache the detection results
(define hardware-info (detect-hardware-capabilities))

;; Accessor functions
(define (get-optimal-num-threads)
  (hash-ref hardware-info 'cores))

(define (has-avx?)
  (hash-ref hardware-info 'avx))

(define (has-sse?)
  (hash-ref hardware-info 'sse))

(define (has-opencl?)
  (hash-ref hardware-info 'opencl))

(define (is-apple-silicon?)
  (hash-ref hardware-info 'apple-silicon))

;; Print hardware information
(define (print-hardware-info)
  (define info hardware-info)
  (printf "======================================\n")
  (printf "Hardware Capabilities for RacoGrad\n")
  (printf "======================================\n")
  (printf "CPU Cores: ~a\n" (hash-ref info 'cores))
  (printf "AVX Support: ~a\n" (hash-ref info 'avx))
  (printf "SSE Support: ~a\n" (hash-ref info 'sse))
  (printf "OpenCL Available: ~a\n" (hash-ref info 'opencl))
  (printf "Apple Silicon: ~a\n" (hash-ref info 'apple-silicon))
  (printf "MLX Support: ~a\n" (hash-ref info 'mlx))
  (printf "CUDA Support: ~a\n" (hash-ref info 'cuda))
  (printf "System Memory: ~a GB\n" (hash-ref info 'memory))
  (printf "--------------------------------------\n")
  (printf "Recommended Configuration:\n")
  (cond
    [(hash-ref info 'mlx)
     (printf "- Use MLX acceleration for Apple Silicon\n")]
    [(hash-ref info 'cuda)
     (printf "- Use CUDA acceleration for NVIDIA GPUs\n")]
    [(hash-ref info 'opencl)
     (printf "- Use OpenCL acceleration for cross-platform GPU support\n")]
    [(hash-ref info 'avx)
     (printf "- Use AVX-optimized functions\n")]
    [(hash-ref info 'sse)
     (printf "- Use SSE-optimized functions\n")]
    [else
     (printf "- Use basic C optimizations\n")])
  (printf "- Use ~a threads for parallel operations\n" (hash-ref info 'cores))
  (printf "======================================\n"))
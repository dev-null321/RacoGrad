#lang racket

(require plot)

(provide plot-training-history)

;; Function to plot training history
;; history: a list of entries in format (epoch loss accuracy val-accuracy)
(define (plot-training-history history [filename "training_history.png"])
  (let* ([epochs (map first history)]
         [losses (map second history)]
         [accuracies (map third history)]
         [val-accuracies (map fourth history)])
    
    ;; Generate the loss plot
    (define loss-plot
      (parameterize ([plot-x-label "Epoch"]
                     [plot-y-label "Loss"]
                     [plot-title "Training Loss"]
                     [plot-font-size 12])
        (plot
         (list
          (points (map vector epochs losses)
                  #:color 'blue
                  #:sym 'circle
                  #:size 6)
          (lines (map vector epochs losses)
                 #:color 'blue
                 #:width 2
                 #:style 'solid
                 #:label "Training Loss"))
         #:x-min 0
         #:y-min 0
         #:width 500
         #:height 300)))
    
    ;; Generate the accuracy plot
    (define accuracy-plot
      (parameterize ([plot-x-label "Epoch"]
                     [plot-y-label "Accuracy (%)"]
                     [plot-title "Training and Validation Accuracy"]
                     [plot-font-size 12])
        (plot
         (list
          (points (map vector epochs accuracies)
                  #:color 'green
                  #:sym 'circle
                  #:size 6)
          (lines (map vector epochs accuracies)
                 #:color 'green
                 #:width 2
                 #:style 'solid
                 #:label "Training Accuracy")
          (points (map vector epochs val-accuracies)
                  #:color 'red
                  #:sym 'triangle
                  #:size 6)
          (lines (map vector epochs val-accuracies)
                 #:color 'red
                 #:width 2
                 #:style 'long-dash
                 #:label "Validation Accuracy"))
         #:x-min 0
         #:y-min 0
         #:y-max 100
         #:width 500
         #:height 300)))
    
    ;; Create a combined plot
    (define combined-plot
      (vl-append 10 loss-plot accuracy-plot))
    
    ;; Save the plots
    (save-plot filename combined-plot 'png)
    
    (printf "Training history plot saved to ~a~n" filename)))

;; Example usage
(module+ main
  ;; Generate sample history data
  (define sample-history
    (for/list ([epoch (in-range 10)])
      (let* ([loss (- 1.0 (* 0.09 epoch))]
             [acc (* 10 epoch)]
             [val-acc (- (* 10 epoch) 5)])
        (list epoch loss acc val-acc))))
  
  ;; Plot the sample data
  (plot-training-history sample-history))
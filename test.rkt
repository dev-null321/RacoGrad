#lang racket

(printf "Starting RacoGrad test script...~n~n")

;; Test MNIST
(printf "Testing MNIST implementation (abbreviated run for quick testing)...~n")
(printf "=============================================================~n")

;; Set MNIST to run just a couple of epochs
(parameterize ([current-command-line-arguments (vector "-m")])
  (dynamic-require "mnist.rkt" #f))

;; Wait a bit
(sleep 2)

;; Test CNN
(printf "~n~nTesting CNN implementation (abbreviated run for quick testing)...~n")
(printf "=================================================================~n")

;; Run CNN with minimal epochs/examples for testing
(parameterize ([current-command-line-arguments (vector "-m")])
  (dynamic-require "CNN.rkt" #f))

(printf "~n~nTests completed!~n")
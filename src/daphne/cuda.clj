(ns daphne.cuda
  "Compile graphical model to CUDA code."
  (:require [clojure.string :as str]))

(def cuda-distributions (slurp "resources/cuda/distributions.cu"))

(def cuda-main (slurp "resources/cuda/main.cu"))

(defn expr->cuda [expr mode observed]
  (cond
    ;; Handle distribution expressions with dynamic mode
    (list? expr)
    (case (first expr)
      ;; Sample and Observe constructs
      sample* (let [[_ dist] expr]
                (expr->cuda dist :sample nil))  ;; Set mode to :sample for sampling
      observe* (let [[_ dist observed] expr]
                 (format "%s"
                         (expr->cuda dist :log-prob observed)))  ;; Set mode to :log-prob for likelihoods

      ;; Distributions
      normal (let [[_ mean stddev] expr]
               (if (= mode :sample)
                 (format "sample_normal(&states[id], %s, %s)" (expr->cuda mean mode nil) (expr->cuda stddev mode nil))
                 (format "log_prob_normal(%s, %s, %s)" observed (expr->cuda mean mode nil) (expr->cuda stddev mode nil))))

      laplace (let [[_ mean scale] expr]
                (if (= mode :sample)
                  (format "sample_laplace(&states[id], %s, %s)" (expr->cuda mean mode nil) (expr->cuda scale mode nil))
                  (format "log_prob_laplace(%s, %s, %s)" observed (expr->cuda mean mode nil) (expr->cuda scale mode nil))))

      uniform (let [[_ lower upper] expr]
                (if (= mode :sample)
                  (format "sample_uniform(&states[id], %s, %s)" (expr->cuda lower mode nil) (expr->cuda upper mode nil))
                  (format "log_prob_uniform(%s, %s, %s)" observed (expr->cuda lower mode nil) (expr->cuda upper mode nil))))
      log-normal (let [[_ mean stddev] expr]
                   (if (= mode :sample)
                     (format "sample_log_normal(&states[id], %s, %s)" (expr->cuda mean mode nil) (expr->cuda stddev mode nil))
                     (format "log_prob_log_normal(%s, %s, %s)" observed (expr->cuda mean mode nil) (expr->cuda stddev mode nil))))
      poisson (let [[_ lambda] expr]
                (if (= mode :sample)
                  (format "sample_poisson(&states[id], %s)" (expr->cuda lambda mode nil))
                  (format "log_prob_poisson(%s, %s)" observed (expr->cuda lambda mode nil))))
      exponential (let [[_ lambda] expr]
                    (if (= mode :sample)
                      (format "sample_exponential(&states[id], %s)" (expr->cuda lambda mode nil))
                      (format "log_prob_exponential(%s, %s)" observed (expr->cuda lambda mode nil))))
      gamma (let [[_ shape scale] expr]
              (if (= mode :sample)
                (format "sample_gamma(&states[id], %s, %s)" (expr->cuda shape mode nil) (expr->cuda scale mode nil))
                (format "log_prob_gamma(%s, %s, %s)" observed (expr->cuda shape mode nil) (expr->cuda scale mode nil))))

      ;; Logical expressions
      if (let [[_ pred then-expr else-expr] expr]
           (format "(%s ? %s : %s)"
                   (expr->cuda pred :predicate nil)
                   (expr->cuda then-expr mode nil)
                   (expr->cuda else-expr mode nil)))
      and (str/join " && " (map #(expr->cuda % :predicate nil) (rest expr)))
      or (str/join " || " (map #(expr->cuda % :predicate nil) (rest expr)))
      not (format "!(%s)" (expr->cuda (second expr) :predicate nil))

      *
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " * " args))

      /
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " / " args))

      +
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " + " args))

      -
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " - " args))

      >
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " > " args))

      <
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " < " args))

      =
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (str/join " == " args))

      tanh
      (let [args (map #(expr->cuda % mode nil) (rest expr))]
        (format "tanh(%s)" (str/join ", " args)))

      ;; Throw an exception for unsupported operators
      (throw (Exception. (str "Unsupported operator: " (first expr)))))

    ;; Symbols represent variables or constants
    (symbol? expr) 
    ;; translate sampleN to samples[id * %s + N]
    (if (str/starts-with? (name expr) "sample")
      (format "samples[id * K + %d]" (dec (Integer. (subs (name expr) 6))))
      (name expr))
    
    (number? expr) (str expr)

    ;; Unsupported types
    :else (throw (Exception. "Unsupported expression type"))))

(defn generate-sample-observe-code [graph]
  (let [samples (filter #(re-find #"sample" (name %)) (:V graph))
        observations (filter #(re-find #"observe" (name %)) (:V graph))
        sample-code (for [sample samples]
                      (let [expr (get-in graph [:P sample])]
                        (format "samples[id * %d + %d] = %s;"
                                (count samples)
                                (Integer. (subs (name sample) 6))
                                (expr->cuda expr :sample nil))))
        observe-code (for [obs observations]
                       (let [expr (get-in graph [:P obs])
                             weight (second (get-in expr [1]))] ;; Extract weight from expressions
                         (format "log_probs[id * %d + %d] = %s;"
                                 (count observations) ;; Store all observations in `results`
                                 (- (Integer. (subs (name obs) 7)) (count samples)) ;; Observation index
                                 (expr->cuda expr :log-prob nil))))]
    ;; Combine sample and observe code
    (concat sample-code observe-code)))

(defn generate-samples [graph]
  ;; Generate `generate_samples` kernel code
  (let [sample-observe-code (generate-sample-observe-code graph)
        kernel-code (str
                     "__global__ void generate_samples(curandState* states, float* samples, float* log_probs, int N, int K) {\n"
                     "  int id = threadIdx.x + blockIdx.x * blockDim.x;\n"
                     "  if (id < N) {\n"
                     (str/join "\n    " sample-observe-code)
                     "\n  }\n"
                     "}\n")]
    kernel-code))

;; Generate CUDA code from graph
(defn generate-cuda [graph]
  (let [num_samples (count (filter #(re-find #"sample" (name %)) (:V graph)))
        num_observations (count (filter #(re-find #"observe" (name %)) (:V graph)))

        kernel-code (generate-samples graph)]

    ;; Write kernels and main function to a file
    (str/join "\n" (concat [cuda-distributions] [kernel-code] [(format cuda-main num_samples num_observations)]))))


(comment
  (def graph
    {:V #{'sample1 'sample2 'observe3 'observe4 'observe5 'observe6 'observe7 'observe8}
     :A
     {'sample1 #{'observe3 'observe4 'observe5 'observe6 'observe7 'observe8}
      'sample2 #{'observe3 'observe4 'observe5 'observe6 'observe7 'observe8}}
     :P
     {'sample1 '(sample* (normal 0.0 10.0))
      'sample2 '(sample* (normal 0.0 10.0))
      'observe3 '(observe* (normal (+ (* sample1 1.0) sample2) 1.0) 2.1)
      'observe4 '(observe* (normal (+ (* sample1 2.0) sample2) 1.0) 3.9)
      'observe5 '(observe* (normal (+ (* sample1 3.0) sample2) 1.0) 5.3)
      'observe6 '(observe* (normal (+ (* sample1 4.0) sample2) 1.0) 7.7)
      'observe7 '(observe* (normal (+ (* sample1 5.0) sample2) 1.0) 10.2)
      'observe8 '(observe* (normal (+ (* sample1 6.0) sample2) 1.0) 12.9)}})

;; Generate the CUDA code fro
  (def cuda-code (generate-cuda graph))

  (println cuda-code)

  (spit "/home/christian/scratch/Development/daphne/generated.cu" cuda-code)

  )

(ns daphne.cuda
  "Compile graphical model to CUDA code."
  (:require [clojure.string :as str]))

(def cuda-distributions (slurp "resources/cuda/distributions.cu"))
(def cuda-main (slurp "resources/cuda/main.cu"))

(defn dispatch-cuda [_env expr]
  (cond
    (list? expr) (first expr)
    (symbol? expr) :symbol
    (number? expr) :number
    :else :unsupported))

(defmulti cuda dispatch-cuda)

(defmethod cuda :symbol
  [{:keys [sample-ids] :as _env} expr]
  (if (str/starts-with? (name expr) "sample")
    (format "samples[id * K + %d]" (sample-ids expr))
    (name expr)))

(defmethod cuda :number
  [_ expr]
  (str expr))

(defmethod cuda 'sample*
  [env [_ dist]]
  (cuda (assoc env :mode :sample) dist))

(defmethod cuda 'observe*
  [env [_ dist observed]]
  (format "%s" (cuda (assoc env :mode :log-prob :observed observed) dist)))

(defmethod cuda 'normal
  [env [_ mean stddev]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_normal(&states[id], %s, %s)"
              (cuda env mean)
              (cuda env stddev))
      (format "log_prob_normal(%s, %s, %s)"
              observed
              (cuda env mean)
              (cuda env stddev)))))

(defmethod cuda 'flip
  [env [_ p]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_bernoulli(&states[id], %s)"
              (cuda env p))
      (format "log_prob_bernoulli(%s, %s)"
              observed
              (cuda env p)))))

(defmethod cuda 'laplace
  [env [_ loc scale]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_laplace(&states[id], %s, %s)"
              (cuda env loc)
              (cuda env scale))
      (format "log_prob_laplace(%s, %s, %s)"
              observed
              (cuda env loc)
              (cuda env scale)))))

(defmethod cuda 'beta
  [env [_ alpha beta]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_beta(&states[id], %s, %s)"
              (cuda env alpha)
              (cuda env beta))
      (format "log_prob_beta(%s, %s, %s)"
              observed
              (cuda env alpha)
              (cuda env beta)))))

(defmethod cuda 'uniform
  [env [_ low high]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_uniform(&states[id], %s, %s)"
              (cuda env low)
              (cuda env high))
      (format "log_prob_uniform(%s, %s, %s)"
              observed
              (cuda env low)
              (cuda env high)))))

(defmethod cuda 'log-normal
  [env [_ mu sigma]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_log_normal(&states[id], %s, %s)"
              (cuda env mu)
              (cuda env sigma))
      (format "log_prob_log_normal(%s, %s, %s)"
              observed
              (cuda env mu)
              (cuda env sigma)))))

(defmethod cuda 'poisson
  [env [_ rate]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_poisson(&states[id], %s)"
              (cuda env rate))
      (format "log_prob_poisson(%s, %s)"
              observed
              (cuda env rate)))))

(defmethod cuda 'exponential
  [env [_ rate]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_exponential(&states[id], %s)"
              (cuda env rate))
      (format "log_prob_exponential(%s, %s)"
              observed
              (cuda env rate)))))

(defmethod cuda 'gamma
  [env [_ shape rate]]
  (let [{:keys [mode observed]} env]
    (if (= mode :sample)
      (format "sample_gamma(&states[id], %s, %s)"
              (cuda env shape)
              (cuda env rate))
      (format "log_prob_gamma(%s, %s, %s)"
              observed
              (cuda env shape)
              (cuda env rate)))))

(defmethod cuda 'if
  [env [_ pred then-expr else-expr]]
  (format "(%s ? %s : %s)"
          (cuda (assoc env :mode :predicate) pred)
          (cuda env then-expr)
          (cuda env else-expr)))

(defmethod cuda 'and
  [env expr]
  (str "(" (str/join " && " (map #(cuda (assoc env :mode :predicate) %) (rest expr))) ")"))

(defmethod cuda 'or
  [env expr]
  (str "(" (str/join " || " (map #(cuda (assoc env :mode :predicate) %) (rest expr))) ")"))

(defmethod cuda 'not
  [env expr]
  (str "(" (format "!(%s)" (cuda (assoc env :mode :predicate) (second expr))) ")"))

(defmethod cuda '=
  [env [_ x y :as expr]]
  (assert (= 3 (count expr)))
  (format "(%s == %s)" (cuda (assoc env :mode :predicate) x) (cuda (assoc env :mode :predicate) y)))

(defmethod cuda '>
  [env [_ x y :as expr]]
  (assert (= 3 (count expr)))
  (format "(%s > %s)" (cuda (assoc env :mode :predicate) x) (cuda (assoc env :mode :predicate) y)))

(defmethod cuda '<
  [env [_ x y :as expr]]
  (assert (= 3 (count expr)))
  (format "(%s < %s)" (cuda (assoc env :mode :predicate) x) (cuda (assoc env :mode :predicate) y)))

(defmethod cuda '*
  [env expr]
  (let [args (map #(cuda env %) (rest expr))]
    (str "(" (str/join " * " args) ")")))

(defmethod cuda '/
  [env expr]
  (let [args (map #(cuda env %) (rest expr))]
    (str "(" (str/join " / " args) ")")))

(defmethod cuda '+
  [env expr]
  (let [args (map #(cuda env %) (rest expr))]
    (str "(" (str/join " + " args) ")")))

(defmethod cuda '-
  [env expr]
  (let [args (map #(cuda env %) (rest expr))]
    (str "(" (str/join " - " args) ")")))

(defmethod cuda 'sqrt
  [env [_ x]]
  (format "sqrtf(%s)" (cuda env x)))

(defmethod cuda 'exp
  [env [_ x]]
  (format "expf(%s)" (cuda env x)))

(defmethod cuda 'log
  [env [_ x]]
  (format "logf(%s)" (cuda env x)))

(defmethod cuda 'tanh
  [env [_ x]]
  (format "tanhf(%s)" (cuda env x)))

(defmethod cuda :unsupported
  [_ _]
  (throw (Exception. "Unsupported expression type")))

;; Main CUDA generation functions
(defn generate-sample-observe-code [graph]
  (let [samples (sort (filter #(re-find #"sample" (name %)) (:V graph)))
        observations (sort (filter #(re-find #"observe" (name %)) (:V graph)))

        sample-ids (into {} (map-indexed (fn [i sample] [sample i]) samples))
        observation-ids (into {} (map-indexed (fn [i obs] [obs i]) observations))

        env {:sample-ids sample-ids
             :observation-ids observation-ids}

        sample-code (for [sample samples]
                      (let [expr (get-in graph [:P sample])]
                        (format "samples[id * %d + %d] = %s;"
                                (count samples)
                                (sample-ids sample)
                                (cuda (merge env {:mode :sample}) expr))))
        observe-code (for [obs observations]
                       (let [expr (get-in graph [:P obs])
                             weight (second (get-in expr [1]))]
                         (format "log_probs[id * %d + %d] = %s;"
                                 (count observations)
                                 (observation-ids obs)
                                 (cuda (merge env {:mode :log-prob :observed weight}) expr))))]
    (concat sample-code observe-code)))

(defn generate-samples [graph]
  (let [sample-observe-code (generate-sample-observe-code graph)
        kernel-code (str
                     "__global__ void generate_samples(curandState* states, float* samples, float* log_probs, int N, int K) {\n"
                     "  int id = threadIdx.x + blockIdx.x * blockDim.x;\n"
                     "  if (id < N) {\n"
                     (str/join "\n    " sample-observe-code)
                     "\n  }\n"
                     "}\n")]
    kernel-code))

(defn generate-cuda [graph]
  (let [num_samples (count (filter #(re-find #"sample" (name %)) (:V graph)))
        num_observations (count (filter #(re-find #"observe" (name %)) (:V graph)))
        kernel-code (generate-samples graph)]
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

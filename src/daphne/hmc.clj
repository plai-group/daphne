(ns daphne.hmc
  (:require [daphne.reverse-diff :refer [fnr normpdf]]
            [daphne.core :refer [code->graph sample-from-prior topo-sort]]
            [daphne.analyze :refer [*bound*]]
            [daphne.free-vars :refer [free-vars]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as op]
            [clojure.core.matrix.linear :as lin]
            [anglican.runtime :refer [observe* sample*
                                      normal uniform-continuous sqrt exp log pow
                                      discrete gamma dirichlet flip covariance
                                      mvn]]))


(def log-optimal-rate (Math/log 0.65))

(def warmup 500)

(defn hmc-step [U gradU M dt nstep [theta0 p0]]
  (let [[theta p] [theta0 p0]
        ;; initial leap-frog symplectic correction
        p (op/- p (op/* (/ dt 2) (apply gradU theta)))
        ;; follow Hamiltonian trajectory
        [theta p]
        (loop [i 0
               [theta p] [theta p]]
          (if (< i nstep)
            (let [gU (apply gradU theta)
                  dtheta (op/* dt (m/mmul M p))
                  theta (op/+ theta dtheta)
                  p (op/- p (op/* dt gU))]
              (recur (inc i)
                     [theta p]))
            [theta p]))
        ;; final leap-frog symplectic correction
        p (op/- p (op/* (/ dt 2) (apply gradU theta)))
        ;; separable Hamiltonian
        H-start (+ (apply U theta0)
                   (/ (m/dot p0 (m/mmul M p0)) 2))
        H-end (+ (apply U theta)
                 ;; Euclidean-Gaussian kinetic energy
                 (/ (m/dot p (m/mmul M p)) 2))
        energy-delta (- H-end H-start)
        log-u (log (rand))
        accepted? (< log-u (- energy-delta))]
    (if accepted?
      [theta p (- energy-delta)]
      [theta0 p0 (- energy-delta)])))

(defn sga [U gradU dt nstep theta0]
  (loop [i 0
         [theta p] [theta0 (m/zero-vector (count theta0))]]
    (let [gU (apply gradU theta)
          dtheta (op/* dt p)
          theta (op/+ theta dtheta)
          p (op/- (op/* 0.9 p) (op/* dt gU))]
      (if (or (< (lin/norm p) 0.00001)
              (= i nstep))
        theta
        (recur (inc i) [theta p])))))

(defn hmc-sample [Urev init]
  (let [U (fn [& args] (first (apply Urev args)))
        gradU (fn [& args] ((second (apply Urev args)) 1.0))
        ;; find typical set (close to MAP estimate)
        ;; this is needed to get a more friendly energy surface for integration
        map-theta (sga U gradU 0.01 1000000 init)]
    ((fn hmc-sample-loop [x total cov-samples
                         [Sigma M]
                         [log-eps d-dt-log-eps log-acceptance-rate]]
       (lazy-seq
        (let [warmup? (<= 0 total (dec warmup))
              eps (Math/exp log-eps)
              L (int (/ 1 eps))
              #_(when (== total warmup)
                  (prn "acc-ratio:" (Math/exp log-acceptance-rate))
                  (prn "eps: " eps)
                  (prn "L: " L)
                  (prn "M-matrix: " M)
                  (prn "warmup state" x))
              [Sigma M] (if (and warmup?
                             (> (count cov-samples) 50))
                      (let [Sigma (covariance (take-last 100 cov-samples))
                            Sigma_inv (m/inverse Sigma)
                            M (op/+ (op/* 0.9 M) (op/* 0.1 Sigma_inv))
                            M (m/div M (m/maximum M))]
                        #_(prn "cov+M" Sigma M)
                        [Sigma M])
                      [Sigma M])

              momentum (sample* (mvn (vec (repeat (count init) 0)) Sigma))
              [x' _ log-accept]
              (hmc-step U gradU M
                        eps
                        L
                        [x momentum])]
          (cons
           x'
           (hmc-sample-loop x'
                            (inc total)
                            (if warmup?
                              (conj cov-samples x')
                              cov-samples)
                            [Sigma M]
                            (if (and warmup? (> log-accept (Math/log 0.01)))
                              ;; use a dynamical system to estimate an adaptation of integration
                              ;; step size through the measured acceptance rate
                              (do
                                [(+ log-eps (* 0.01 d-dt-log-eps))
                                 (+ (* 0.1 d-dt-log-eps)
                                    (- log-acceptance-rate
                                       log-optimal-rate))
                                 (+ (* 0.9 log-acceptance-rate)
                                    (* 0.1 log-accept))])
                              [(if warmup?
                                 (- log-eps 1)
                                 log-eps) ;; TODO step back to previous
                               0
                               log-acceptance-rate]))))))
     map-theta 1.0 []
     [(m/identity-matrix (count init))
      (m/identity-matrix (count init))]
     [(Math/log 0.001) 0 log-optimal-rate])))

(def dist->pdf {'normal 'normpdf})

(defn hmc
  ([code]
   (let [[rho graph return] (code->graph code)
         initial-X (->> [rho graph return]
                      sample-from-prior
                      butlast
                      (into {}))
         samples (->> (topo-sort graph)
                      (filter #(re-find #"sample\d+" (name %))))
         return-fn (eval `(fn [~(vec samples)] ~return))]
     (->>
      (hmc samples initial-X [rho graph return])
      (map return-fn))))
  ([samples initial-X [rho graph return]]
   (let [observes (->> (topo-sort graph)
                       (filter #(re-find #"observe\d+" (name %))))
         Urev (binding [*ns* (find-ns 'daphne.hmc)]
               (eval
                `(fnr ~(vec samples)
                      (-
                       (+ ~@(concat
                             (map (fn [o]
                                    (let [[_ [dist & params] value] (o (:P graph))]
                                      (concat [(dist->pdf dist)] [value] params ))) observes)
                             (map (fn [s]
                                    (let [[_ [dist & params]] (s (:P graph))]
                                      (concat [(dist->pdf dist)] [s] params)))
                                  samples)))))))]
     (drop warmup
           (hmc-sample Urev (vec (map initial-X samples)))))))


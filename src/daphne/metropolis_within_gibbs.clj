(ns daphne.metropolis-within-gibbs
  (:require [daphne.core :refer [code->graph sample-from-prior topo-sort]]
            [daphne.analyze :refer [*bound*]]
            [daphne.free-vars :refer [free-vars]]
            [anglican.runtime :refer :all
             #_[observe* sample*
               normal uniform-continuous sqrt exp
               discrete gamma dirichlet flip]]))

(defn build-proposal-map [P]
  (->> P
     (filter (fn [[k v]] (re-find #"sample\d+" (name k))))
     (map (fn [[k v]] [(keyword k) (list 'fn [] (second v))]))
   (into {})))

(defn sample->observe [[sym exp]]
  (if (re-find #"sample\d+" (name sym))
    (list 'observe* (second exp) sym)
    exp))

(defn build-log-likelihood-map [P]
  (->> P
       (map (fn [[k v]]
              [(keyword k) (list 'fn [] (sample->observe [k v]))]))
       (into {})))

(defn build-log-likelihoods [graph]
  `(~'fn [{:syms ~(vec (:V graph))} x#]
    ((get ~(build-log-likelihood-map (:P graph)) (keyword x#)))))

(defn build-proposals [graph]
  `(~'fn [{:syms ~(vec (:V graph))} x#]
    ((get ~(build-proposal-map (:P graph)) (keyword x#)))))

(defn accept-markov-blanket [{:keys [proposals likelihoods graph]} x X' X]
  (let [q (proposals X x)
        q' (proposals X' x)
        log-alpha (- (observe* q' (X x))
                     (observe* q (X' x)))
        V_x (conj ((:A graph) x) x)]
    (exp
     (reduce (fn [log-alpha v]
               (+ log-alpha
                  (likelihoods X' v)
                  (- (likelihoods X v))))
             log-alpha
             V_x))))

(defn tol? [a b] (< (Math/abs (- a b)) 1e-10))

(defn gibbs-substep [{:keys [proposals graph] :as params} X x]
  (let [X' (assoc X x (sample* (proposals X x)))
        alpha-markov (accept-markov-blanket params x X' X)
        u (sample* (uniform-continuous 0 1))]
    (if (< u alpha-markov)
      X'
      X)))

(defn gibbs-step [params X]
  (reduce (partial gibbs-substep params)
          X
          (:var-order params)))

(defn metropolis-within-gibbs
  ([code]
   (let [[rho graph return] (code->graph code)
         initial-X (->> [rho graph return]
                      sample-from-prior
                      butlast
                      (into {}))
         return-fn (eval `(fn [{:syms ~(vec (:V graph))}]
                            ~return))]
     (->>
      (metropolis-within-gibbs initial-X [rho graph return])
      (map return-fn))))
  ([initial-X [rho graph return]]
   (let [var-order (->> (topo-sort graph)
                      (filter #(re-find #"sample\d+" (name %))))
         ;; we build fast pre-compiled routines here
         proposals (binding [*ns* (find-ns 'daphne.metropolis-within-gibbs)]
                     (eval (build-proposals graph)))
         likelihoods (binding [*ns* (find-ns 'daphne.metropolis-within-gibbs)]
                       (eval (build-log-likelihoods graph)))
         params {:likelihoods likelihoods
                 :proposals proposals
                 :graph graph
                 :var-order var-order}]
     ((fn metropolis-gibbs-internal [X]
        (lazy-seq
         (let [new-X (gibbs-step params X)]
           (cons X
                 (metropolis-gibbs-internal new-X)))))
      initial-X))))




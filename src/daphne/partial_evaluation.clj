(ns daphne.partial-evaluation
  (:require [daphne.substitute :refer [substitute]]
            [daphne.symbolic-simplify :refer [symbolic-simplify]]
            [daphne.desugar-let :refer [desugar-let]]
            [daphne.desugar :refer [desugar]]
            [clojure.core.memoize :as memoize]
            [daphne.gensym :as gensym]
            [anglican.runtime]))



(defn dispatch-partial-evaluation [exp]
  (cond (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'defn))
        :defn

        (and (list? exp)
             (= (first exp) 'loop))
        :loop

        (and (list? exp)
             (= (first exp) 'foreach))
        :foreach

        (and (list? exp)
             (not ((ns-map 'clojure.core)
                   (first exp))))
        :unapplication

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (list? exp)
        :application

        (seq? exp)
        :seq

        (map? exp)
        :map

        (vector? exp)
        :vector

        :else
        :unrelated))


(defmulti partial-evaluation dispatch-partial-evaluation)


(defn value? [x]
  (if (or (seq? x)
          (vector? x)
          (map? x)
          (set? x))
    (not (some symbol? (flatten x)))
    (not (symbol? x))))


(def mem-eval (memoize/lu eval :lu/threshold 2048))

(defn rand-symbol? [x]
  (and (symbol? x)
       #_(not (*bound* x))
       (#{\0 \1 \2 \3 \4 \5 \6 \7 \8 \9} (last (name x)))))

(defmethod partial-evaluation :let
  [exp]
  (let [[_ bindings & body] exp
        evaled-bindings' (map (fn [[s v]]
                                #_(when-not (symbol? s)
                                  (throw (ex-info "Not a symbol."
                                                  {:symbol s :value v})))
                                [s (partial-evaluation v)])
                              (partition 2 bindings))
        evaled-bindings (vec (apply concat evaled-bindings'))
        new-let
        (apply list
               (concat (list 'let evaled-bindings)
                       (map (fn [exp]
                              (let [sub-exp (reduce (fn [exp [s v]]
                                                      (if (value? v)
                                                        (substitute exp s v)
                                                        exp))
                                                    exp
                                                    (partition 2 evaled-bindings))]
                                (partial-evaluation sub-exp)))
                            body)))]
    (if (some rand-symbol? (map second evaled-bindings'))
      new-let
      (try
        (mem-eval new-let)
        (catch Exception _
          new-let)))))

(defmethod partial-evaluation :defn
  [exp]
  (let [[_ name bindings & body] exp]
    (apply list (concat (list 'defn name bindings)
                        (map partial-evaluation body)))))

(defmethod partial-evaluation :loop
  [exp]
  (let [[_ counter & rest] exp]
    (apply list 'loop (partial-evaluation counter) rest)))

(defmethod partial-evaluation :foreach
  [exp]
  (let [[_ counter & rest] exp]
    (apply list 'foreach (partial-evaluation counter) rest)))

(defmethod partial-evaluation :unapplication
  [exp]
  (apply list (conj (map partial-evaluation (rest exp))
                    (first exp))))


(defmethod partial-evaluation :if
  [exp]
  (let [[_ cond then else] exp
        eval-cond (partial-evaluation cond)]
    (cond (true? eval-cond)
          (partial-evaluation then)

          (false? eval-cond)
          (partial-evaluation else)

          :else
          (list 'if eval-cond
                (partial-evaluation then)
                (partial-evaluation else)))))



(defmethod partial-evaluation :application
  [exp]
  (if (some rand-symbol? (flatten exp))
    (apply list
           (conj (map partial-evaluation (rest exp))
                 (first exp)))
    (try
      (mem-eval exp)
      (catch Exception _
        (apply list
               (conj (map partial-evaluation (rest exp))
                     (first exp)))))))

(defmethod partial-evaluation :seq
  [exp]
  (map partial-evaluation exp))

(defmethod partial-evaluation :vector
  [exp]
  (mapv partial-evaluation exp))

(defmethod partial-evaluation :unrelated
  [exp]
  exp)

(defmethod partial-evaluation :map
  [exp]
  (->> (interleave (map partial-evaluation (keys exp))
                 (map partial-evaluation (vals exp)))
     (partition 2)
     (map vec)
     (into {})))


#_(defn safe-desugar [x]
  (try (desugar x)
       (catch Exception e
         (when-let [m (ex-data e)]
           (prn "desugaring failed" m))
         x)))

(def ^:dynamic *show-exp* true)

(def ^:dynamic *show-exp* false)


(defn raw-fixed-point-simplify [exp]
    (let [gensyms (atom (range))]
      (binding [gensym/*my-gensym* (fn [s]
                                     (let [f (first @gensyms)]
                                       (swap! gensyms rest)
                                       (symbol (str s f))))]
        #_(when (< (rand) 0.001)
          (println "Trying to simplify" exp))
        (loop [exp exp
               i 0]
          (when (> i 100)
            (println "Warning, 100 times simplified on" exp))
          #_(reset! gensyms (range))
          #_(when *show-exp*
            (println "fixed point search" exp))
          (let [new-exp (-> exp
                         desugar-let
                         partial-evaluation
                         symbolic-simplify
                         desugar)]
          (if (= new-exp exp)
            exp
            (recur new-exp (inc i))))))))


(def fixed-point-simplify (memoize/lu raw-fixed-point-simplify :lu/threshold 2048))

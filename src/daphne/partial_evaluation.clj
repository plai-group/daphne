(ns daphne.partial-evaluation
  (:require [daphne.substitute :refer [substitute]]
            [daphne.symbolic-simplify :refer [mem-symbolic-simplify]]
            [daphne.desugar-let :refer [mem-desugar-let]]
            [daphne.desugar :refer [mem-desugar]]
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

(def mem-partial-evaluation (memoize/lu partial-evaluation :lu/threshold 100000))

(defn value? [x]
  (if (or (seq? x)
          (vector? x)
          (map? x)
          (set? x))
    (not (some symbol? (flatten x)))
    (not (symbol? x))))

(def mem-eval (memoize/lu eval :lu/threshold 10000))

(defn rand-symbol? [x]
  (and (symbol? x)
       (.startsWith (name x) "sample_")
       #_(not (*bound* x))
       #_(#{\0 \1 \2 \3 \4 \5 \6 \7 \8 \9} (last (name x)))))

(defmethod partial-evaluation :let
  [exp]
  (let [[_ bindings & body] exp
        evaled-bindings' (map (fn [[s v]]
                                #_(when-not (symbol? s)
                                  (throw (ex-info "Not a symbol."
                                                  {:symbol s :value v})))
                                [s (mem-partial-evaluation v)])
                              (partition 2 bindings))
        evaled-bindings (vec (apply concat evaled-bindings'))
        new-body (map (fn [exp]
                        (let [sub-exp (reduce (fn [exp [s v]]
                                                (if (value? v)
                                                  (substitute exp s v)
                                                  exp))
                                              exp
                                              (partition 2 evaled-bindings))]
                          (mem-partial-evaluation sub-exp)))
                      body)
        new-let
        (apply list
               (concat (list 'let evaled-bindings)
                       new-body))]
    (if (some #(not (value? %)) (flatten (map second evaled-bindings')))
      new-let
      (try
        (mem-eval new-let)
        (catch Exception _
          (if (value? body)
            (last body)
            new-let))))))

(defmethod partial-evaluation :defn
  [exp]
  (let [[_ name bindings & body] exp]
    (apply list (concat (list 'defn name bindings)
                        (map mem-partial-evaluation body)))))

(defmethod partial-evaluation :loop
  [exp]
  (let [[_ counter & rest] exp]
    (apply list 'loop (mem-partial-evaluation counter) rest)))

(defmethod partial-evaluation :foreach
  [exp]
  (let [[_ counter & rest] exp]
    (apply list 'foreach (mem-partial-evaluation counter) rest)))

(defmethod partial-evaluation :unapplication
  [exp]
  (apply list (conj (map mem-partial-evaluation (rest exp))
                    (first exp))))

(defmethod partial-evaluation :if
  [exp]
  (let [[_ cond then else] exp
        eval-cond (mem-partial-evaluation cond)]
    (cond (true? eval-cond)
          (mem-partial-evaluation then)

          (false? eval-cond)
          (mem-partial-evaluation else)

          :else
          (list 'if eval-cond
                (mem-partial-evaluation then)
                (mem-partial-evaluation else)))))

(defmethod partial-evaluation :application
  [exp]
  (if (some rand-symbol? (flatten exp))
    (apply list
           (conj (map mem-partial-evaluation (rest exp))
                 (first exp)))
    (try
      (mem-eval exp)
      (catch Exception _
        (apply list
               (conj (map mem-partial-evaluation (rest exp))
                     (first exp)))))))

(defmethod partial-evaluation :seq
  [exp]
  (map mem-partial-evaluation exp))

(defmethod partial-evaluation :vector
  [exp]
  (mapv mem-partial-evaluation exp))

(defmethod partial-evaluation :unrelated
  [exp]
  exp)

(defmethod partial-evaluation :map
  [exp]
  (->> (interleave (map mem-partial-evaluation (keys exp))
                 (map mem-partial-evaluation (vals exp)))
     (partition 2)
     (map vec)
     (into {})))

(defn raw-fixed-point-simplify [exp]
    (let [gensyms (atom (range))]
      (binding [gensym/*my-gensym* (fn [s]
                                     (let [f (first @gensyms)]
                                       (swap! gensyms rest)
                                       (symbol (str s f))))]
        (loop [exp exp
               i 0]
          (when (= i 100)
            (println "Warning, 100 times simplified on" exp))
          #_(println "fixed point search" i)
          (let [new-exp (-> exp
                           mem-desugar-let
                           mem-partial-evaluation
                           mem-symbolic-simplify
                           mem-desugar)]
          (if (= new-exp exp)
            exp
            (recur new-exp (inc i))))))))

(def fixed-point-simplify (memoize/lu raw-fixed-point-simplify :lu/threshold 10000))

(ns daphne.desugar-hoppl)


(defn dispatch-desugar-hoppl
  [exp]
  (cond (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'fn))
        :fn

        (and (list? exp)
             (= (first exp) 'defn))
        :defn

        (map? exp)
        :map

        (list? exp)
        :list

        (seq? exp)
        :seq

        (vector? exp)
        :vector

        :else :unrelated))


(defmulti desugar-hoppl dispatch-desugar-hoppl)

(defmethod desugar-hoppl :let
  [exp]
  (let [[_ bindings & body] exp]
    (let [[_ bindings' & body] exp
          bindings             (partition 2 bindings')]
      (apply list (apply list 'fn (mapv first bindings) body)
             (mapv second bindings)))))

(defmethod desugar-hoppl :fn [exp]
  (let [[op args & body] exp]
    (apply list op args (map desugar-hoppl body))))

(defmethod desugar-hoppl :defn [exp]
  (let [[op name args & body] exp]
    (apply list op name args (map desugar-hoppl body))))

(defmethod desugar-hoppl :map [exp]
  (conj (map (fn [[k v]]
               [(desugar-hoppl k)
                (desugar-hoppl v)])
             exp)
        'hash-map))

(defmethod desugar-hoppl :list [exp]
  (apply list (map #(desugar-hoppl %) exp)))

(defmethod desugar-hoppl :seq [exp]
  (map #(desugar-hoppl %) exp))

(defmethod desugar-hoppl :vector [exp]
  (apply list
         (conj
          (map #(desugar-hoppl %) exp)
          'vector)))

(defmethod desugar-hoppl :unrelated [exp]
  exp)


(comment
  (desugar-hoppl '(+ 1 2)) ;; => (+ 1 2)


  (desugar-hoppl '(let [x 1] (+ x 2))) ;; => ((fn [x] (+ x 2)) 1)

  )

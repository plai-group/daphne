(ns daphne.desugar-datastructures)

(defn dispatch-desugar-datastructures
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


(defmulti desugar-datastructures dispatch-desugar-datastructures)

(defmethod desugar-datastructures :let
  [exp]
  (let [[_ bindings & body] exp]
    (apply list
           'let (vec
                 (mapcat (fn [[k v]]
                           [k (desugar-datastructures v)])
                         (partition 2 bindings)))
           (map desugar-datastructures body))))

(defmethod desugar-datastructures :fn [exp]
  (let [[op args & body] exp]
    (apply list op args (map desugar-datastructures body))))

(defmethod desugar-datastructures :defn [exp]
  (let [[op name args & body] exp]
    (apply list op name args (map desugar-datastructures body))))

(defmethod desugar-datastructures :map [exp]
  (conj (map (fn [[k v]]
               [(desugar-datastructures k)
                (desugar-datastructures v)])
             exp)
        'hash-map))

(defmethod desugar-datastructures :list [exp]
  (apply list (map #(desugar-datastructures %) exp)))

(defmethod desugar-datastructures :seq [exp]
  (map #(desugar-datastructures %) exp))

(defmethod desugar-datastructures :vector [exp]
  (apply list
         (conj
          (map #(desugar-datastructures %) exp)
          'vector)))

(defmethod desugar-datastructures :unrelated [exp]
  exp)


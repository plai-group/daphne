(ns daphne.desugar-hoppl
  (:require [daphne.gensym :refer [*my-gensym*]]))


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

        (and (list? exp)
             (= (first exp) 'loop))
        :loop

        (and (list? exp)
             (= (first exp) 'foreach))
        :foreach

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
    ((fn expand-bindings [[f & r]]
       (let [[b v] f]
         (if f
           (list
            (list 'fn #_(*my-gensym* "let") [b]
                  (expand-bindings r))
            (desugar-hoppl v))
           ((fn expand-body [[f & r]]
              (if-not (empty? r)
                (list 'let [(*my-gensym* "dontcare") (desugar-hoppl f)]
                      (expand-body r))
                (desugar-hoppl f)))
            body))))
     (partition 2 bindings))))

(defmethod desugar-hoppl :fn [exp]
  (let [[op #_name args & body] exp]
    (assert (= op 'fn))
    #_(assert (symbol? name))
    (assert (vector? args))
    (apply list 'fn #_name args (map desugar-hoppl body))))

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

(defn desugar-defn [exp]
  (let [[op name args & body] exp]
    (assert (= op 'defn))
    (assert (symbol? name))
    (assert (vector? args))
    [name (apply list 'fn #_name args body)]))

;; not used atm. because of global let binding for defns
(defmethod desugar-hoppl :defn [exp]
  (let [[op name args & body] exp]
    (apply list op name args (map desugar-hoppl body))))


(defmethod desugar-hoppl :loop [exp]
  (let [[_ c e f & es] exp
        as (map (fn [_] (*my-gensym*)) es)]
    (list 'let
          (vec
           (concat
            ['bound c
             'initial-value e]
            (partition 2 (interleave as es))
            ['g (list 'fn #_(*my-gensym* "loop") ['i 'w] (apply list f 'i 'w as))]
            ))
          (list 'loop-helper 0 'bound 'initial-value 'g))))

(comment
  (defmethod desugar-hoppl :foreach [exp]
    (let [[_ c e f & es] exp
          as             (map (fn [_] (*my-gensym*)) es)]
      (list 'let
            (vec
             (concat
              ['bound c
               'initial-value e]
              (partition 2 (interleave as es))
              ['g (list 'fn (*my-gensym* "loop") ['i 'w] (apply list f 'i 'w as))]
              ))
            (list 'loop-helper 0 'bound 'initial-value 'g)))))

(def preamble '[#_(defn loop-helper [i c v g]
                  (if (= i c)
                    v
                    (loop-helper (+ i 1) c (g i v) g)))])

(defn desugar-hoppl-global [code]
  (let [defns (concat preamble (butlast code))
        main (last code)]
    (desugar-hoppl
     (list 'let (vec (mapcat desugar-defn defns))
           main))))


(comment
  (desugar-hoppl '(+ 1 2)) ;; => (+ 1 2)


  (eval (desugar-hoppl '(let [x 1 y 3] (+ x y 1)))) ;; => ((fn [x] (+ x 2)) 1)

  (eval (desugar-hoppl-global '[(defn add [a b] (+ a b)) (let [a (add 2 3)] (- a 1))]))

  (eval (desugar-hoppl-global '[(loop 3 0 (fn [i c] (+ c 1)))]))


  )

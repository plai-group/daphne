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
    (assert (even? (count bindings)) "Let requires an even number of bindings.")
    ((fn expand-bindings [[f & r]]
       (let [[b v] f]
         (if f
           (list
            (list 'fn #_(*my-gensym* "let") [b]
                  (expand-bindings r))
            (desugar-hoppl v))
           ((fn expand-body [[f & r]]
              (if-not (empty? r)
                (list (list 'fn [(*my-gensym* "dontcare")]
                            (expand-body r))
                      (desugar-hoppl f))
                #_(list 'let [(*my-gensym* "dontcare") (desugar-hoppl f)]
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
  (apply list
         (conj (mapcat (fn [[k v]]
                         [(desugar-hoppl k)
                          (desugar-hoppl v)])
                       exp)
               'hash-map)))

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

(defn extend-call-sites [name exp]
  (cond (list? exp)
        (let [[f & r] exp]
          (if (= f name)
            (apply list (concat [f] r [f]))
            ;; shadowed by lambda argument => skip extending
            (if (and (= f 'fn) (some #(= name %) (first r)))
              exp
              (apply list (map (partial extend-call-sites name) exp)))))

        (seq? exp)
        (doall (map (partial extend-call-sites name) exp))

        (vector? exp)
        (mapv (partial extend-call-sites name) exp)

        (map? exp)
        (into {}
              (map (fn [[k v]] [(extend-call-sites name k)
                               (extend-call-sites name v)])
                   exp))

        :else exp))

(defn desugar-defn [exp]
  (let [[op name args & body] exp]
    (assert (= op 'defn))
    (assert (symbol? name))
    (assert (vector? args))
    (let [aug-f (apply list 'fn #_name (conj args name)
                       (extend-call-sites name (desugar-hoppl body)))]
      [name (list 'fn args
                  (apply list (concat [aug-f] args [aug-f])))])))

;; not used atm. because of global let binding for defns
(defmethod desugar-hoppl :defn [exp]
  (let [[op name args & body] exp]
    (assert (symbol? name))
    (assert (vector? args))
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

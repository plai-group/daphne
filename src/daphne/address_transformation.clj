(ns daphne.address-transformation
  "See Section 6.2 in Introduction to Probabilistic Programming."
  (:require [daphne.gensym :refer [*my-gensym*]]))


(defn dispatch-address-trafo
  [exp _]
  (cond (and (list? exp)
             (= (first exp) 'fn))
        :fn

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (map? exp)
        :map

        (list? exp)
        :list

        (seq? exp)
        :seq

        (vector? exp)
        :vector

        :else :unrelated))


(defmulti address-trafo dispatch-address-trafo)

;; we do not know about primitives functions here and therefore also create new
;; symbols for those invocations
(defmethod address-trafo :fn [exp alpha]
  (let [[op #_name args & body] exp]
    (assert (= op 'fn))
    #_(assert (symbol? name))
    (assert (vector? args))
    (apply list 'fn #_name (vec (concat [alpha] args))
           (map #(address-trafo % alpha) body))))

(defmethod address-trafo :if [exp alpha]
  (apply list (map #(address-trafo % alpha) exp)))

(defmethod address-trafo :map [exp alpha]
  (into {} (map (fn [[k v]]
                  [(address-trafo k alpha)
                   (address-trafo v alpha)])
                exp)))

(defmethod address-trafo :list [exp alpha]
  (let [[f & args] exp
        c (*my-gensym* "addr")]
    (apply list (address-trafo f alpha) (list 'push-address alpha c)
           (map #(address-trafo % alpha) args))))

(defmethod address-trafo :seq [exp alpha]
  (doall (map #(address-trafo % alpha) exp)))

(defmethod address-trafo :vector [exp alpha]
  (apply list
         (conj
          (map #(address-trafo % alpha) exp)
          'vector)))

(defmethod address-trafo :unrelated [exp alpha]
  exp)







(comment
  (address-trafo '(+ 1 2) []) ;; => (+ 1 2)

  (address-trafo '((fn [x] (sample (normal (+ x 1) 1))) 2) 'alpha5)


  (eval (address-trafo '(let [x 1 y 3] (+ x y 1)))) ;; => ((fn [x] (+ x 2)) 1)


  )

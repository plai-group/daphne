(ns daphne.hoppl-cps
  (:require [daphne.gensym :refer [*my-gensym*]]
            [clojure.walk :as walk]))


(defn dispatch-hoppl-cps [exp _]
  (cond (and (list? exp)
             (= (first exp) 'fn))
        :fn

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (list? exp)
        :list

        :else :unrelated))

(defmulti hoppl-cps dispatch-hoppl-cps)

(defmethod hoppl-cps :fn [exp k]
  (let [[_ args body] exp
        k'             (*my-gensym* "k")]
    (list k
          (list 'fn (conj args k')
                (hoppl-cps body k')))))

(defmethod hoppl-cps :if [exp k]
  (let [[_ pred then else] exp
        k' (*my-gensym* "cps")]
    (hoppl-cps pred (list 'fn [k']
                          (list 'if k'
                                (hoppl-cps then k)
                                (hoppl-cps else k))))))

(defn invert-control-flow
  [exp k collected]
  (cond (and (list? exp)
             (= (first exp) 'fn))
        (hoppl-cps exp k)

        (and (list? exp)
             (= (first exp) 'if))
        (let [exp (let [[_ pred then else] exp
                        k'                 (*my-gensym* "ifcps")]
                    (with-meta 
                      (hoppl-cps pred (list 'fn [k']
                                            (list 'if k'
                                                  (hoppl-cps then k)
                                                  (hoppl-cps else k))))
                      {:ifcps true}))
              v   (*my-gensym* "cps")]
          (swap! collected conj [v exp])
          v)

        (list? exp)
        (let [exp (doall
                   ;; undo call of continuation by unwrapping second, because we
                   ;; are not returning yet, but passing the values to a function
                   (map #(let [res (invert-control-flow % k collected)]
                          (if (and (list? %) (= 'if (first %)))
                            res
                            (second res))) exp))
              v   (*my-gensym* "cps")]
          (swap! collected conj [v exp])
          (list k v))

        :else (list k exp)))

(defn nest-lambdas [control-flow k]
  (let [[[k' exp] & r] control-flow]
    (apply list
           (if (:ifcps (meta exp))
             ;; TODO this does not respect bindings that could shadow the
             ;; continuation
             (walk/postwalk (fn [x]
                              (if (= x k)
                                (if (seq r)
                                  (list 'fn [k']
                                        (nest-lambdas r k))
                                  k)
                                x))
                            exp)
             (concat exp [(if (seq r)
                            (list 'fn [k']
                                  (nest-lambdas r k))
                            k)])))))

(defmethod hoppl-cps :list [exp k]
  (nest-lambdas (let [collected (atom [])]
                  (invert-control-flow exp k collected)
                  @collected)
                k))

(defmethod hoppl-cps :unrelated [exp k]
  (list k exp))



(comment

  (nest-lambdas
   (let [collected (atom [])]
     (invert-control-flow '(sqrt (+ (* x x) (* y y))) 'k123 collected)
     @collected)
   'k123)


  (nest-lambdas
   (let [collected (atom [])]
     (invert-control-flow '(+ 3 ((fn [x] x) 2)) 'k123 collected)
     @collected)

   'k123)

  (hoppl-cps '(fn [x] x) 'k123)


  (hoppl-cps '(fn [x] (+ 1 x)) 'k123)


  (hoppl-cps '(sqrt (+ (* x x) (* y y))) 'k123)

  (hoppl-cps '(+ 3 ((fn [x] x) 2)) 'k123)

  (hoppl-cps '((fn [x] (+ 1 x)) 2) 'k123)

  (nest-lambdas
   (invert-control-flow '((fn [x] (+ 1 x)) 2) 'k123)

   'k123)




  (let [k123 (fn [x] (println "result" x))
        + (fn [a b k] (k (clojure.core/+ a b)))]
    ((fn [x k29096] (k29096 x)) 2 (fn [cps29097] (+ 3 cps29097 k123)))) 

  ;; correct
  (hoppl-cps '(if (= 2 2) (+ 2 1) 3) 'k123)

  ;; correct
  (hoppl-cps '(if true (+ 2 1) 3) 'k123)

  ;; correct
  (hoppl-cps '(if true (+ ((fn [x] (+ 1 x)) 2) 1) 3) 'k123)

  ((fn [cps30331]
     (if cps30331
       ((fn [x k30332] (+ 1 x k30332)) 2 (fn [cps30334] (+ cps30334 1 k123)))
       (k123 3)))
   true)

  ;; faulty
  (hoppl-cps
   '((fn [x] (+ 2 (if (even? x) (+ 1 2) 3))) 5)
   'k123)

  (hoppl-cps '((fn [x] (+ 1 x)) 2) 'k123)

  (defn +& [a b k]
    (k (+ a b)))

  ;; CPS examples

  42 ;; =>

  (hoppl-cps
   '((fn [a b] (+ a b)) 1 2))



  (+& 1 2 (fn [v] (println "current value:" v)))



  (hoppl-cps '(fn [a b]
                  (+ a b)))

  )

